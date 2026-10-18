"""Static anti-gaming policy: what a candidate's source may not do, checked before it is compiled.

A candidate can compile, score well and even return the right answer while the work is done by
something other than its kernel. After AKO4X's benchmark rule ("write your own kernel") and its
TRAPS catalogs, this rejects:
  * library delegation: a host function computing the problem's operator with torch on the
    inputs it was given (torch.matmul, `@`, x.sum(), torch.linalg.cholesky, ...). Torch as glue
    (empty, zeros, reshape, strides, reductions over the kernel's own partial results) is fine.
    Inputs are followed through views, casts and calls into helper functions;
  * a kernel that is never launched: some public host function must launch a @triton.jit
    kernel, and only those functions are offered to the harness as entry points;
  * monkeypatching: assignment to attributes of torch, triton or any imported name, setattr,
    delattr, and getattr on a module with a computed name;
  * imports a kernel has no use for (os, sys, subprocess, socket, pickle, numpy, ...) and calls
    to open, exec, eval, compile, __import__, globals;
  * definitions of the harness's own names (reference, make_inputs, ...).
A module-level container written by a host function, a functools cache, `global`, and
data_ptr()/id() lookups are warnings only: an honest kernel may cache compiled configurations,
and the dynamic audit in harness.py catches a cache that serves stale results.
"""
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from harness import RESERVED_NAMES

ALLOWED_IMPORTS = frozenset(("torch", "triton", "math", "typing", "functools", "itertools", "operator",
                             "dataclasses", "enum", "collections", "numbers", "__future__"))
FORBIDDEN_CALLS = frozenset(("open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                             "setattr", "delattr", "breakpoint", "input"))
# Calls that compute each problem's operator. On the inputs of a host function they are delegation.
DELEGATION = {
    "matmul_v2": frozenset(("matmul", "mm", "bmm", "addmm", "baddbmm", "addbmm", "addmv", "mv", "einsum",
                            "linear", "tensordot", "dot", "inner", "outer", "chain_matmul", "multi_dot")),
    "vectorsum_v2": frozenset(("sum", "nansum", "cumsum", "mean", "nanmean", "dot", "inner", "einsum", "norm",
                               "vector_norm", "trapz", "trapezoid", "logsumexp")),
    "cholesky": frozenset(("cholesky", "cholesky_ex", "cholesky_inverse", "cholesky_solve", "solve_triangular",
                           "triangular_solve", "lu", "lu_factor", "lu_factor_ex", "ldl_factor", "ldl_factor_ex",
                           "eigh", "eig", "qr", "svd", "inv", "inv_ex", "solve", "solve_ex")),
    "trimul_alphafold3": frozenset(("matmul", "bmm", "einsum", "tensordot")),
}
MATMUL_PROBLEMS = frozenset(("matmul_v2", "trimul_alphafold3"))
# Tensor methods and attributes whose result is still the input: views, casts, copies
PASSTHROUGH = frozenset(("view", "reshape", "contiguous", "flatten", "t", "transpose", "permute", "to", "float",
                         "half", "bfloat16", "double", "clone", "detach", "expand", "squeeze", "unsqueeze",
                         "view_as", "reshape_as", "T", "mT", "type", "cpu", "cuda", "narrow"))
LIBRARY_ROOTS = ("torch", "F")
MUTATORS = frozenset(("append", "extend", "insert", "update", "setdefault", "add", "__setitem__"))


@dataclass
class PolicyResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    entries: List[str] = field(default_factory=list)  # public host functions that launch a kernel


def _is_kernel(fn: ast.FunctionDef) -> bool:
    return any(k in ast.unparse(d) for d in fn.decorator_list for k in ("jit", "autotune", "heuristics"))


def _dotted(node: ast.AST) -> Optional[str]:
    """'torch.linalg.cholesky' for a chain of attributes on a name, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id] + parts[::-1])
    return None


def _root(node: ast.AST) -> Optional[str]:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _derived(node: ast.AST, tainted: Set[str]) -> bool:
    """Whether `node` is one of the tainted names, or a view, cast or slice of one."""
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Subscript):
        return _derived(node.value, tainted)
    if isinstance(node, ast.Attribute):
        return node.attr in PASSTHROUGH and _derived(node.value, tainted)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PASSTHROUGH:
        if _derived(node.func.value, tainted):
            return True  # x.contiguous()
        return (_dotted(node.func.value) or "").split(".")[0] in LIBRARY_ROOTS and bool(node.args) \
            and _derived(node.args[0], tainted)  # torch.flatten(x)
    return False


def _taint(fn: ast.FunctionDef, params: Set[str]) -> Set[str]:
    """Names in `fn` that hold (views of) the given parameters."""
    tainted = set(params)
    for _ in range(2):  # a second pass catches aliases assigned after first use
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                pairs = []
                for target in node.targets:
                    if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                        pairs += list(zip(target.elts, node.value.elts))
                    else:
                        pairs.append((target, node.value))
                for target, value in pairs:
                    if isinstance(target, ast.Name) and _derived(value, tainted):
                        tainted.add(target.id)
    return tainted


def _params(fn: ast.FunctionDef) -> List[str]:
    return [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]


def _launches(fn: ast.FunctionDef, kernels: Set[str]) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Subscript) and _root(node.func.value) in kernels:
                return True  # kernel[grid](...)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "run" and _root(node.func.value) in kernels:
                return True  # kernel.run(..., grid=...)
    return False


def _calls(fn: ast.FunctionDef, names: Set[str]) -> Set[str]:
    return {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in names}


def check(code: str, problem_id: str) -> PolicyResult:
    result = PolicyResult()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result.errors.append(f"syntax error at line {e.lineno}")
        return result

    # Imports: the only modules a kernel and its host wrapper need
    imported: Set[str] = set()
    library_functions: Dict[str, str] = {}  # `from torch import matmul` -> {"matmul": "torch.matmul"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    result.errors.append(f"imports {alias.name}, which a kernel has no use for")
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module.split(".")[0] not in ALLOWED_IMPORTS:
                result.errors.append(f"imports from {'.' * node.level}{module}, which a kernel has no use for")
            for alias in node.names:
                imported.add(alias.asname or alias.name)
                if module.split(".")[0] == "torch":
                    library_functions[alias.asname or alias.name] = f"{module}.{alias.name}"

    # Monkeypatching, forbidden calls, dynamic lookups on modules
    for node in ast.walk(tree):
        targets = (node.targets if isinstance(node, (ast.Assign, ast.Delete))
                   else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        flat = []
        for target in targets:
            flat += target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
        for target in flat:
            if isinstance(target, (ast.Attribute, ast.Subscript)) and _root(target) in imported:
                result.errors.append(f"monkeypatching: assigns to {ast.unparse(target)}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                result.errors.append(f"calls {node.func.id}(), which a kernel has no use for")
            elif node.func.id == "getattr" and node.args and _root(node.args[0]) in imported:
                name = node.args[1] if len(node.args) > 1 else None
                if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                    result.errors.append(f"looks up a computed attribute name on {ast.unparse(node.args[0])}, "
                                         "which hides what is called")
                elif name.value in DELEGATION.get(problem_id, ()):
                    result.errors.append(f"library delegation: getattr({ast.unparse(node.args[0])}, {name.value!r})")

    top = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    kernels = {name for name, fn in top.items() if _is_kernel(fn)}
    hosts = {name: fn for name, fn in top.items() if name not in kernels}
    for node in tree.body:
        names = ([node.name] if isinstance(node, ast.FunctionDef) else
                 [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else [])
        for name in names:
            if name in RESERVED_NAMES:
                result.errors.append(f"defines `{name}`, a name the harness reserves")

    # Which host functions launch a kernel, directly or through other host functions
    launching = {name for name, fn in hosts.items() if _launches(fn, kernels)}
    changed = True
    while changed:
        changed = False
        for name, fn in hosts.items():
            if name not in launching and _calls(fn, launching):
                launching.add(name)
                changed = True
    result.entries = sorted(n for n in launching if not n.startswith("_") and n not in RESERVED_NAMES)
    if not result.entries:
        result.errors.append("no public host function launches a @triton.jit kernel")

    # Delegation: follow the inputs of every public host function into its helpers
    tainted: Dict[str, Set[str]] = {name: set(_params(fn)) if not name.startswith("_") else set()
                                    for name, fn in hosts.items()}
    changed = True
    while changed:
        changed = False
        for name, fn in hosts.items():
            names = _taint(fn, tainted[name])
            for call in ast.walk(fn):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in hosts):
                    continue
                params = _params(hosts[call.func.id])
                passed = {params[i] for i, arg in enumerate(call.args) if i < len(params) and _derived(arg, names)}
                passed |= {kw.arg for kw in call.keywords if kw.arg in params and _derived(kw.value, names)}
                if not passed <= tainted[call.func.id]:
                    tainted[call.func.id] |= passed
                    changed = True
    operator = DELEGATION.get(problem_id, frozenset())
    for name, fn in hosts.items():
        names = _taint(fn, tainted[name])
        for node in ast.walk(fn):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult) and problem_id in MATMUL_PROBLEMS \
                    and (_derived(node.left, names) or _derived(node.right, names)):
                result.errors.append(f"library delegation: `{ast.unparse(node)[:60]}` in {name}() multiplies its "
                                     "inputs with torch instead of a kernel")
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            args = list(node.args) + [kw.value for kw in node.keywords]
            if isinstance(func, ast.Attribute) and func.attr in operator:
                library = (_dotted(func.value) or "").split(".")[0] in LIBRARY_ROOTS
                if _derived(func.value, names) or (library and any(_derived(a, names) for a in args)):
                    result.errors.append(f"library delegation: `{ast.unparse(node)[:60]}` in {name}() computes the "
                                         "operator with torch instead of a kernel")
            elif isinstance(func, ast.Name) and func.id in library_functions \
                    and library_functions[func.id].split(".")[-1] in operator and any(_derived(a, names) for a in args):
                result.errors.append(f"library delegation: `{ast.unparse(node)[:60]}` in {name}()")

    # Hidden state: warnings, confirmed or cleared by the dynamic audit
    containers = {t.id for n in tree.body if isinstance(n, ast.Assign)
                  and isinstance(n.value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.Call))
                  and not (isinstance(n.value, ast.Call) and _dotted(n.value.func) not in
                           ("dict", "list", "set", "collections.defaultdict", "defaultdict", "OrderedDict",
                            "collections.OrderedDict", "deque", "collections.deque"))
                  for t in n.targets if isinstance(t, ast.Name)}
    for name, fn in hosts.items():
        for node in ast.walk(fn):
            if isinstance(node, ast.Global):
                result.warnings.append(f"{name}() uses `global {', '.join(node.names)}`")
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) and _root(t) in containers
                                                      for t in node.targets):
                result.warnings.append(f"{name}() writes the module-level container `{_root(node.targets[0])}`")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in MUTATORS and _root(node.func.value) in containers:
                    result.warnings.append(f"{name}() writes the module-level container `{_root(node.func.value)}`")
                elif node.func.attr == "data_ptr":
                    result.warnings.append(f"{name}() keys on tensor addresses (data_ptr())")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "id":
                result.warnings.append(f"{name}() keys on object identity (id())")
        for d in fn.decorator_list:
            if "cache" in ast.unparse(d):
                result.warnings.append(f"{name}() is cached with {ast.unparse(d)}")
    result.errors = list(dict.fromkeys(result.errors))
    result.warnings = list(dict.fromkeys(result.warnings))
    return result
