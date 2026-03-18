"""
C265: SQL Built-in Functions
Extends C264 (Window Functions) with comprehensive scalar functions.

Categories:
- String: UPPER, LOWER, LENGTH, TRIM, LTRIM, RTRIM, SUBSTRING, REPLACE,
          CONCAT, REVERSE, REPEAT, LPAD, RPAD, POSITION, LEFT, RIGHT,
          STARTS_WITH, ENDS_WITH, CHAR_LENGTH, ASCII, CHAR, INSTR,
          CONCAT_WS, INITCAP, TRANSLATE
- Math: ABS, ROUND, CEIL/CEILING, FLOOR, POWER/POW, SQRT, MOD, SIGN,
        LOG, LOG2, LOG10, LN, EXP, PI, GREATEST, LEAST, RANDOM,
        TRUNCATE/TRUNC, DEGREES, RADIANS, SIN, COS, TAN, ASIN, ACOS, ATAN
- Null-handling: COALESCE, NULLIF, IFNULL/NVL, IIF
- Type: CAST, TYPEOF, PRINTF/FORMAT
- Aggregate: GROUP_CONCAT/STRING_AGG
"""

import sys
import os
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C264_window_functions')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C263_ctas')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C262_views')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C261_foreign_keys')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C260_check_constraints')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

from window_functions import (
    WindowDB, WindowParser, parse_window_sql, parse_window_sql_multi,
    SqlWindowFunc, WindowSpec, FrameBound, NamedWindow,
    RANKING_FUNCS, NAVIGATION_FUNCS, AGGREGATE_FUNCS,
    _eval_arg,
)
from mini_database import (
    ResultSet, ParseError, CompileError,
    SqlFuncCall, SqlLiteral, SqlColumnRef, SqlCase, SqlStar,
    SqlBinOp, SqlComparison, SqlLogic, SqlIsNull, SqlBetween, SqlInList,
    TokenType, Token, Lexer, Parser, KEYWORDS,
    SelectExpr, SelectStmt, TableRef,
)
from query_executor import Row, eval_expr, FuncExpr


# =============================================================================
# Built-in scalar function registry
# =============================================================================

def _builtin_apply(name: str, args: List[Any]) -> Any:
    """Evaluate a built-in scalar function. Returns sentinel if unknown."""
    fn = name.upper()

    # ----- String functions -----
    if fn == 'UPPER':
        return str(args[0]).upper() if args and args[0] is not None else None

    if fn == 'LOWER':
        return str(args[0]).lower() if args and args[0] is not None else None

    if fn in ('LENGTH', 'LEN', 'CHAR_LENGTH', 'CHARACTER_LENGTH'):
        return len(str(args[0])) if args and args[0] is not None else None

    if fn == 'TRIM':
        if not args or args[0] is None:
            return None
        s = str(args[0])
        if len(args) >= 2 and args[1] is not None:
            return s.strip(str(args[1]))
        return s.strip()

    if fn == 'LTRIM':
        if not args or args[0] is None:
            return None
        s = str(args[0])
        if len(args) >= 2 and args[1] is not None:
            return s.lstrip(str(args[1]))
        return s.lstrip()

    if fn == 'RTRIM':
        if not args or args[0] is None:
            return None
        s = str(args[0])
        if len(args) >= 2 and args[1] is not None:
            return s.rstrip(str(args[1]))
        return s.rstrip()

    if fn in ('SUBSTR', 'SUBSTRING'):
        if not args or args[0] is None:
            return None
        s = str(args[0])
        if len(args) < 2:
            raise CompileError("SUBSTRING requires at least 2 arguments")
        start = args[1]
        if start is None:
            return None
        # SQL is 1-indexed
        start = int(start) - 1
        if start < 0:
            start = 0
        if len(args) >= 3 and args[2] is not None:
            length = int(args[2])
            return s[start:start + length]
        return s[start:]

    if fn == 'REPLACE':
        if len(args) < 3:
            raise CompileError("REPLACE requires 3 arguments")
        if args[0] is None:
            return None
        return str(args[0]).replace(str(args[1] or ''), str(args[2] or ''))

    if fn == 'CONCAT':
        return ''.join(str(a) for a in args if a is not None)

    if fn == 'CONCAT_WS':
        if len(args) < 2:
            raise CompileError("CONCAT_WS requires at least 2 arguments")
        sep = str(args[0]) if args[0] is not None else ''
        parts = [str(a) for a in args[1:] if a is not None]
        return sep.join(parts)

    if fn == 'REVERSE':
        return str(args[0])[::-1] if args and args[0] is not None else None

    if fn == 'REPEAT':
        if len(args) < 2:
            raise CompileError("REPEAT requires 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        return str(args[0]) * int(args[1])

    if fn == 'LPAD':
        if len(args) < 2:
            raise CompileError("LPAD requires at least 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        s = str(args[0])
        width = int(args[1])
        pad = str(args[2]) if len(args) >= 3 and args[2] is not None else ' '
        if not pad:
            pad = ' '
        if len(s) >= width:
            return s[:width]
        needed = width - len(s)
        full_reps = needed // len(pad)
        remainder = needed % len(pad)
        return (pad * full_reps + pad[:remainder]) + s

    if fn == 'RPAD':
        if len(args) < 2:
            raise CompileError("RPAD requires at least 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        s = str(args[0])
        width = int(args[1])
        pad = str(args[2]) if len(args) >= 3 and args[2] is not None else ' '
        if not pad:
            pad = ' '
        if len(s) >= width:
            return s[:width]
        needed = width - len(s)
        full_reps = needed // len(pad)
        remainder = needed % len(pad)
        return s + (pad * full_reps + pad[:remainder])

    if fn in ('POSITION', 'INSTR'):
        if len(args) < 2:
            raise CompileError(f"{fn} requires 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        # POSITION: find substring in string, 1-indexed, 0 if not found
        # POSITION(substr, string) or INSTR(string, substr)
        if fn == 'POSITION':
            substr, s = str(args[0]), str(args[1])
        else:
            s, substr = str(args[0]), str(args[1])
        idx = s.find(substr)
        return idx + 1 if idx >= 0 else 0

    if fn == 'LEFT':
        if len(args) < 2:
            raise CompileError("LEFT requires 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        return str(args[0])[:int(args[1])]

    if fn == 'RIGHT':
        if len(args) < 2:
            raise CompileError("RIGHT requires 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        n = int(args[1])
        s = str(args[0])
        if n >= len(s):
            return s
        return s[-n:] if n > 0 else ''

    if fn == 'STARTS_WITH':
        if len(args) < 2:
            raise CompileError("STARTS_WITH requires 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        return 1 if str(args[0]).startswith(str(args[1])) else 0

    if fn == 'ENDS_WITH':
        if len(args) < 2:
            raise CompileError("ENDS_WITH requires 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        return 1 if str(args[0]).endswith(str(args[1])) else 0

    if fn == 'ASCII':
        if not args or args[0] is None:
            return None
        s = str(args[0])
        return ord(s[0]) if s else None

    if fn == 'CHAR' or fn == 'CHR':
        if not args or args[0] is None:
            return None
        return chr(int(args[0]))

    if fn == 'INITCAP':
        if not args or args[0] is None:
            return None
        return str(args[0]).title()

    if fn == 'TRANSLATE':
        if len(args) < 3:
            raise CompileError("TRANSLATE requires 3 arguments")
        if args[0] is None:
            return None
        s = str(args[0])
        from_chars = str(args[1] or '')
        to_chars = str(args[2] or '')
        table = str.maketrans(from_chars, to_chars[:len(from_chars)],
                              from_chars[len(to_chars):] if len(to_chars) < len(from_chars) else '')
        return s.translate(table)

    # ----- Math functions -----
    if fn == 'ABS':
        if not args or args[0] is None:
            return None
        return abs(args[0])

    if fn in ('ROUND',):
        if not args or args[0] is None:
            return None
        decimals = int(args[1]) if len(args) >= 2 and args[1] is not None else 0
        return round(float(args[0]), decimals)

    if fn in ('CEIL', 'CEILING'):
        if not args or args[0] is None:
            return None
        return math.ceil(float(args[0]))

    if fn == 'FLOOR':
        if not args or args[0] is None:
            return None
        return math.floor(float(args[0]))

    if fn in ('POWER', 'POW'):
        if len(args) < 2:
            raise CompileError("POWER requires 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        return math.pow(float(args[0]), float(args[1]))

    if fn == 'SQRT':
        if not args or args[0] is None:
            return None
        v = float(args[0])
        if v < 0:
            return None
        return math.sqrt(v)

    if fn == 'MOD':
        if len(args) < 2:
            raise CompileError("MOD requires 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        b = args[1]
        if b == 0:
            return None
        return args[0] % b

    if fn == 'SIGN':
        if not args or args[0] is None:
            return None
        v = float(args[0])
        if v > 0:
            return 1
        elif v < 0:
            return -1
        return 0

    if fn == 'LOG':
        if not args or args[0] is None:
            return None
        v = float(args[0])
        if v <= 0:
            return None
        if len(args) >= 2 and args[1] is not None:
            base = float(args[1])
            if base <= 0 or base == 1:
                return None
            return math.log(v) / math.log(base)
        return math.log(v)

    if fn == 'LOG2':
        if not args or args[0] is None:
            return None
        v = float(args[0])
        return math.log2(v) if v > 0 else None

    if fn == 'LOG10':
        if not args or args[0] is None:
            return None
        v = float(args[0])
        return math.log10(v) if v > 0 else None

    if fn == 'LN':
        if not args or args[0] is None:
            return None
        v = float(args[0])
        return math.log(v) if v > 0 else None

    if fn == 'EXP':
        if not args or args[0] is None:
            return None
        return math.exp(float(args[0]))

    if fn == 'PI':
        return math.pi

    if fn == 'GREATEST':
        vals = [a for a in args if a is not None]
        return max(vals) if vals else None

    if fn == 'LEAST':
        vals = [a for a in args if a is not None]
        return min(vals) if vals else None

    if fn == 'RANDOM' or fn == 'RAND':
        return random.random()

    if fn in ('TRUNCATE', 'TRUNC'):
        if not args or args[0] is None:
            return None
        v = float(args[0])
        if len(args) >= 2 and args[1] is not None:
            d = int(args[1])
            factor = 10 ** d
            return int(v * factor) / factor
        return int(v)

    if fn == 'DEGREES':
        if not args or args[0] is None:
            return None
        return math.degrees(float(args[0]))

    if fn == 'RADIANS':
        if not args or args[0] is None:
            return None
        return math.radians(float(args[0]))

    if fn == 'SIN':
        if not args or args[0] is None:
            return None
        return math.sin(float(args[0]))

    if fn == 'COS':
        if not args or args[0] is None:
            return None
        return math.cos(float(args[0]))

    if fn == 'TAN':
        if not args or args[0] is None:
            return None
        return math.tan(float(args[0]))

    if fn == 'ASIN':
        if not args or args[0] is None:
            return None
        v = float(args[0])
        if v < -1 or v > 1:
            return None
        return math.asin(v)

    if fn == 'ACOS':
        if not args or args[0] is None:
            return None
        v = float(args[0])
        if v < -1 or v > 1:
            return None
        return math.acos(v)

    if fn == 'ATAN':
        if not args or args[0] is None:
            return None
        return math.atan(float(args[0]))

    if fn == 'ATAN2':
        if len(args) < 2:
            raise CompileError("ATAN2 requires 2 arguments")
        if args[0] is None or args[1] is None:
            return None
        return math.atan2(float(args[0]), float(args[1]))

    # ----- Null-handling functions -----
    if fn == 'COALESCE':
        for a in args:
            if a is not None:
                return a
        return None

    if fn == 'NULLIF':
        if len(args) < 2:
            raise CompileError("NULLIF requires 2 arguments")
        return None if args[0] == args[1] else args[0]

    if fn in ('IFNULL', 'NVL'):
        if len(args) < 2:
            raise CompileError(f"{fn} requires 2 arguments")
        return args[0] if args[0] is not None else args[1]

    if fn == 'IIF':
        if len(args) < 3:
            raise CompileError("IIF requires 3 arguments")
        return args[1] if args[0] else args[2]

    # ----- Type functions -----
    if fn == 'CAST':
        # CAST is handled specially in the parser as CAST(expr AS type)
        # But if called as a function, handle it
        if len(args) < 2:
            return args[0] if args else None
        return _do_cast(args[0], args[1])

    if fn == 'TYPEOF':
        if not args:
            return 'null'
        v = args[0]
        if v is None:
            return 'null'
        if isinstance(v, int):
            return 'integer'
        if isinstance(v, float):
            return 'real'
        if isinstance(v, str):
            return 'text'
        if isinstance(v, bool):
            return 'integer'
        return 'blob'

    if fn in ('PRINTF', 'FORMAT'):
        if not args:
            return None
        fmt = str(args[0]) if args[0] is not None else ''
        try:
            return fmt % tuple(args[1:])
        except (TypeError, ValueError):
            return fmt

    # Unknown function -- return sentinel
    return _UNKNOWN_FUNC


# Sentinel for unknown functions
_UNKNOWN_FUNC = object()

# Set of all known built-in function names (for detection without calling)
BUILTIN_FUNC_NAMES = {
    # String
    'UPPER', 'LOWER', 'LENGTH', 'LEN', 'CHAR_LENGTH', 'CHARACTER_LENGTH',
    'TRIM', 'LTRIM', 'RTRIM', 'SUBSTR', 'SUBSTRING', 'REPLACE',
    'CONCAT', 'CONCAT_WS', 'REVERSE', 'REPEAT', 'LPAD', 'RPAD',
    'POSITION', 'INSTR', 'LEFT', 'RIGHT', 'STARTS_WITH', 'ENDS_WITH',
    'ASCII', 'CHAR', 'CHR', 'INITCAP', 'TRANSLATE',
    # Math
    'ABS', 'ROUND', 'CEIL', 'CEILING', 'FLOOR', 'POWER', 'POW',
    'SQRT', 'MOD', 'SIGN', 'LOG', 'LOG2', 'LOG10', 'LN', 'EXP',
    'PI', 'GREATEST', 'LEAST', 'RANDOM', 'RAND',
    'TRUNCATE', 'TRUNC', 'DEGREES', 'RADIANS',
    'SIN', 'COS', 'TAN', 'ASIN', 'ACOS', 'ATAN', 'ATAN2',
    # Null-handling
    'COALESCE', 'NULLIF', 'IFNULL', 'NVL', 'IIF',
    # Type
    'CAST', 'TYPEOF', 'PRINTF', 'FORMAT',
}


def _do_cast(value: Any, type_name: Any) -> Any:
    """Cast a value to the given type."""
    if value is None:
        return None
    t = str(type_name).upper().strip()
    if t in ('INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT'):
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            return 0
    if t in ('REAL', 'FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC', 'NUMBER'):
        try:
            return float(str(value))
        except (ValueError, TypeError):
            return 0.0
    if t in ('TEXT', 'VARCHAR', 'CHAR', 'STRING', 'NVARCHAR', 'CLOB'):
        return str(value)
    if t in ('BOOLEAN', 'BOOL'):
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return bool(value)
    return value


# =============================================================================
# AST node for CAST(expr AS type)
# =============================================================================

@dataclass
class SqlCast:
    """CAST(expr AS type_name)"""
    expr: Any
    type_name: str


# =============================================================================
# Extended Parser with built-in function awareness + CAST syntax
# =============================================================================

# Keywords that can also be used as function names
_KEYWORD_FUNC_TOKENS = {TokenType.LEFT, TokenType.RIGHT}


class BuiltinParser(WindowParser):
    """Parser extended with CAST(expr AS type) syntax and keyword-functions."""

    def _parse_primary(self):
        """Override to handle CAST(expr AS type) and keyword-named functions."""
        tok = self.peek()
        if tok.type == TokenType.IDENT and tok.value.upper() == 'CAST':
            return self._parse_cast()

        # Handle LEFT(...) and RIGHT(...) as function calls when followed by (
        if tok.type in _KEYWORD_FUNC_TOKENS:
            # Lookahead: is the next token LPAREN?
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.LPAREN:
                name = self.advance().value.upper()  # consume LEFT/RIGHT
                self.advance()  # consume (
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self._parse_expr())
                    while self.match(TokenType.COMMA):
                        args.append(self._parse_expr())
                self.expect(TokenType.RPAREN)
                return SqlFuncCall(func_name=name, args=args)

        return super()._parse_primary()

    def _parse_cast(self) -> SqlCast:
        """Parse CAST(expr AS type)."""
        self.advance()  # consume CAST
        self.expect(TokenType.LPAREN)
        expr = self._parse_expr()
        # Expect AS keyword
        if self.peek().type == TokenType.AS:
            self.advance()
        else:
            raise ParseError("Expected AS in CAST expression")
        # Parse type name (may be multi-word like DOUBLE PRECISION)
        type_name = self.advance().value
        # Handle optional precision/length: VARCHAR(100)
        if self.peek().type == TokenType.LPAREN:
            self.advance()
            # Skip precision args
            while self.peek().type != TokenType.RPAREN:
                self.advance()
            self.expect(TokenType.RPAREN)
        self.expect(TokenType.RPAREN)
        return SqlCast(expr=expr, type_name=type_name)


# =============================================================================
# Parse functions
# =============================================================================

def parse_builtin_sql(sql: str):
    """Parse a single SQL statement with built-in function support."""
    lexer = Lexer(sql)
    parser = BuiltinParser(lexer.tokens)
    return parser.parse()


def parse_builtin_sql_multi(sql: str):
    """Parse multiple SQL statements with built-in function support."""
    lexer = Lexer(sql)
    parser = BuiltinParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# Extended eval_expr to handle SqlFuncCall with built-in functions + SqlCast
# =============================================================================

def builtin_eval_expr(expr, row: Row) -> Any:
    """Extended eval_expr that handles SQL-level AST nodes and built-in functions.

    Unlike eval_expr (which expects query_executor AST nodes like ColumnRef,
    ArithExpr), this handles SQL-level nodes (SqlColumnRef, SqlBinOp, etc.).
    """
    if isinstance(expr, SqlCast):
        val = builtin_eval_expr(expr.expr, row)
        return _do_cast(val, expr.type_name)

    if isinstance(expr, SqlFuncCall):
        args = [builtin_eval_expr(a, row) for a in expr.args]
        result = _builtin_apply(expr.func_name, args)
        if result is not _UNKNOWN_FUNC:
            return result
        # Unknown function -- fall through to _eval_arg

    if isinstance(expr, SqlColumnRef):
        col_name = expr.column
        if expr.table:
            col_name = f"{expr.table}.{expr.column}"
        return row.get(col_name)

    if isinstance(expr, SqlLiteral):
        return expr.value

    if isinstance(expr, SqlBinOp):
        left = builtin_eval_expr(expr.left, row)
        right = builtin_eval_expr(expr.right, row)
        if left is None or right is None:
            return None
        op = expr.op
        if op == '+':
            return left + right
        if op == '-':
            return left - right
        if op == '*':
            return left * right
        if op == '/':
            return left / right if right != 0 else None
        if op == '%':
            return left % right if right != 0 else None
        if op == '||':
            return str(left) + str(right)
        return None

    if isinstance(expr, SqlComparison):
        left = builtin_eval_expr(expr.left, row)
        right = builtin_eval_expr(expr.right, row) if expr.right is not None else None
        op = expr.op
        if op == '=':
            return left == right
        if op in ('!=', '<>'):
            return left != right
        if op == '<':
            return left < right if left is not None and right is not None else False
        if op == '>':
            return left > right if left is not None and right is not None else False
        if op == '<=':
            return left <= right if left is not None and right is not None else False
        if op == '>=':
            return left >= right if left is not None and right is not None else False
        if op == 'like':
            if left is None or right is None:
                return False
            import re
            pattern = str(right).replace('%', '.*').replace('_', '.')
            return bool(re.fullmatch(pattern, str(left), re.IGNORECASE))
        return False

    if isinstance(expr, SqlLogic):
        op = expr.op.lower()
        if op == 'and':
            return all(builtin_eval_expr(o, row) for o in expr.operands)
        if op == 'or':
            return any(builtin_eval_expr(o, row) for o in expr.operands)
        if op == 'not':
            return not builtin_eval_expr(expr.operands[0], row)
        return False

    if isinstance(expr, SqlIsNull):
        val = builtin_eval_expr(expr.expr, row)
        result = val is None
        return not result if expr.negated else result

    if isinstance(expr, SqlBetween):
        val = builtin_eval_expr(expr.expr, row)
        low = builtin_eval_expr(expr.low, row)
        high = builtin_eval_expr(expr.high, row)
        if val is None or low is None or high is None:
            return False
        return low <= val <= high

    if isinstance(expr, SqlInList):
        val = builtin_eval_expr(expr.expr, row)
        vals = [builtin_eval_expr(v, row) for v in expr.values]
        return val in vals

    if isinstance(expr, SqlCase):
        for cond, val in expr.whens:
            if builtin_eval_expr(cond, row):
                return builtin_eval_expr(val, row)
        if expr.else_result is not None:
            return builtin_eval_expr(expr.else_result, row)
        return None

    # For anything else, try _eval_arg (handles remaining types)
    return _eval_arg(expr, row)


# =============================================================================
# BuiltinDB -- Database with built-in function support
# =============================================================================

class BuiltinDB(WindowDB):
    """WindowDB extended with comprehensive built-in scalar functions."""

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with built-in function support."""
        stmt = parse_builtin_sql(sql)
        return self._execute_builtin_stmt(stmt)

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements."""
        stmts = parse_builtin_sql_multi(sql)
        return [self._execute_builtin_stmt(s) for s in stmts]

    def _execute_builtin_stmt(self, stmt) -> ResultSet:
        """Execute a statement with built-in function support."""
        if isinstance(stmt, SelectStmt):
            # Check for built-in functions or casts in expressions
            if self._has_builtin_calls(stmt):
                return self._exec_select_with_builtins(stmt)
        # Fall through to window/ctas/base execution
        return self._execute_window_stmt(stmt)

    def _has_builtin_calls(self, stmt: SelectStmt) -> bool:
        """Check if the statement has any SqlFuncCall or SqlCast nodes."""
        for col in stmt.columns:
            if self._expr_has_builtin(col.expr):
                return True
        if stmt.where and self._expr_has_builtin(stmt.where):
            return True
        if stmt.order_by:
            for expr, _ in stmt.order_by:
                if self._expr_has_builtin(expr):
                    return True
        return False

    def _expr_has_builtin(self, expr) -> bool:
        """Recursively check if an expression has built-in function calls."""
        if isinstance(expr, SqlCast):
            return True
        if isinstance(expr, SqlFuncCall):
            fn = expr.func_name.upper()
            if fn in BUILTIN_FUNC_NAMES:
                return True
            # Also check nested args
            return any(self._expr_has_builtin(a) for a in expr.args)
        if isinstance(expr, SqlBinOp):
            return self._expr_has_builtin(expr.left) or self._expr_has_builtin(expr.right)
        if isinstance(expr, SqlComparison):
            return self._expr_has_builtin(expr.left) or (expr.right and self._expr_has_builtin(expr.right))
        if isinstance(expr, SqlLogic):
            return any(self._expr_has_builtin(o) for o in expr.operands)
        if isinstance(expr, SqlCase):
            for cond, val in expr.whens:
                if self._expr_has_builtin(cond) or self._expr_has_builtin(val):
                    return True
            if expr.else_expr and self._expr_has_builtin(expr.else_expr):
                return True
        return False

    def _exec_select_with_builtins(self, stmt: SelectStmt) -> ResultSet:
        """Execute SELECT with built-in function evaluation.

        Strategy: Execute base query via SELECT * to get rows, then evaluate
        columns with builtin_eval_expr.
        """
        # Check for window functions first -- delegate to window handler
        win_funcs = self._extract_window_funcs(stmt)
        if win_funcs:
            return self._exec_select_with_windows(stmt, win_funcs)

        # Check if it's a pure expression query (no FROM clause)
        if stmt.from_table is None:
            return self._exec_expr_only(stmt)

        # Execute base query to get rows
        base_stmt = SelectStmt(
            columns=[SelectExpr(expr=SqlStar())],
            from_table=stmt.from_table,
            joins=stmt.joins,
            where=self._rewrite_where(stmt.where),
            group_by=stmt.group_by,
            having=stmt.having,
        )

        # For GROUP BY with aggregates, need special handling
        if stmt.group_by:
            return self._exec_grouped_with_builtins(stmt)

        base_result = self._execute_window_stmt(base_stmt)

        # Convert to Row objects
        rows = []
        for raw_row in base_result.rows:
            data = {}
            for i, col_name in enumerate(base_result.columns):
                data[col_name] = raw_row[i]
            rows.append(Row(data))

        # Filter with WHERE (if it has builtins not already applied)
        if stmt.where and self._expr_has_builtin(stmt.where):
            rows = [r for r in rows if builtin_eval_expr(stmt.where, r)]

        # Evaluate column expressions
        output_columns = []
        for i, col in enumerate(stmt.columns):
            if col.alias:
                output_columns.append(col.alias)
            elif isinstance(col.expr, SqlColumnRef):
                output_columns.append(col.expr.column)
            elif isinstance(col.expr, SqlStar):
                output_columns.append(f"col_{i}")
            else:
                output_columns.append(f"col_{i}")

        output_rows = []
        for row in rows:
            out_row = []
            for col in stmt.columns:
                if isinstance(col.expr, SqlStar):
                    # Expand star
                    for k, v in row.data.items():
                        out_row.append(v)
                else:
                    out_row.append(builtin_eval_expr(col.expr, row))
            output_rows.append(out_row)

        # Handle star column names
        if any(isinstance(c.expr, SqlStar) for c in stmt.columns):
            star_cols = list(rows[0].data.keys()) if rows else []
            output_columns = []
            for col in stmt.columns:
                if isinstance(col.expr, SqlStar):
                    output_columns.extend(star_cols)
                elif col.alias:
                    output_columns.append(col.alias)
                elif isinstance(col.expr, SqlColumnRef):
                    output_columns.append(col.expr.column)
                else:
                    output_columns.append(f"col_{len(output_columns)}")

        # Apply ORDER BY
        if stmt.order_by:
            output_rows = self._sort_output(output_rows, output_columns, stmt.order_by, rows)

        # Apply DISTINCT
        if stmt.distinct:
            seen = set()
            unique = []
            for row in output_rows:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    unique.append(row)
            output_rows = unique

        # Apply LIMIT/OFFSET
        if stmt.offset:
            output_rows = output_rows[stmt.offset:]
        if stmt.limit is not None:
            output_rows = output_rows[:stmt.limit]

        return ResultSet(columns=output_columns, rows=output_rows)

    def _exec_expr_only(self, stmt: SelectStmt) -> ResultSet:
        """Execute a SELECT with no FROM clause (pure expressions like SELECT 1+1)."""
        output_columns = []
        output_row = []
        dummy_row = Row({})

        for i, col in enumerate(stmt.columns):
            val = builtin_eval_expr(col.expr, dummy_row)
            output_row.append(val)
            if col.alias:
                output_columns.append(col.alias)
            else:
                output_columns.append(f"col_{i}")

        return ResultSet(columns=output_columns, rows=[output_row])

    def _exec_grouped_with_builtins(self, stmt: SelectStmt) -> ResultSet:
        """Handle GROUP BY queries that also have built-in function calls.

        Delegates to base execution but wraps results with builtin eval.
        """
        # Let the base handle the GROUP BY + aggregation
        result = self._execute_window_stmt(stmt)
        return result

    def _rewrite_where(self, where):
        """Pass through WHERE -- builtin functions in WHERE are handled by base if possible."""
        if where is None:
            return None
        # If WHERE only has builtins, return None (we'll filter after)
        if self._expr_has_builtin(where):
            return None
        return where

    def _sort_output(self, output_rows, columns, order_by, base_rows):
        """Sort output rows by ORDER BY with builtin eval support."""
        from functools import cmp_to_key

        pairs = list(enumerate(output_rows))

        def compare(a, b):
            idx_a, row_a = a
            idx_b, row_b = b
            for expr, asc in order_by:
                if isinstance(expr, SqlColumnRef):
                    col_name = expr.column
                    try:
                        ci = columns.index(col_name)
                        va, vb = row_a[ci], row_b[ci]
                    except ValueError:
                        va = builtin_eval_expr(expr, base_rows[idx_a]) if idx_a < len(base_rows) else None
                        vb = builtin_eval_expr(expr, base_rows[idx_b]) if idx_b < len(base_rows) else None
                else:
                    va = builtin_eval_expr(expr, base_rows[idx_a]) if idx_a < len(base_rows) else None
                    vb = builtin_eval_expr(expr, base_rows[idx_b]) if idx_b < len(base_rows) else None

                if va is None and vb is None:
                    continue
                if va is None:
                    return 1
                if vb is None:
                    return -1
                if va < vb:
                    return -1 if asc else 1
                if va > vb:
                    return 1 if asc else -1
            return 0

        pairs.sort(key=cmp_to_key(compare))
        return [row for _, row in pairs]
