"""
C269: SQL String Concatenation (|| operator)
Extends C268 (Set Operations) with the || string concatenation operator.

The || operator:
- Concatenates two values as strings: 'foo' || 'bar' -> 'foobar'
- Converts non-string operands: 42 || ' items' -> '42 items'
- NULL propagation: NULL || 'foo' -> NULL
- Left-associative: 'a' || 'b' || 'c' -> 'abc'
- Precedence: between comparison and arithmetic (lower than +/-)
- Works in SELECT, WHERE, ORDER BY, GROUP BY, HAVING, CASE, CTEs, subqueries
- Chainable with other operators: first_name || ' ' || last_name

Implementation notes:
- Lexer: CONCAT token type added to C247 (recognizes ||)
- Parser: _parse_concat() method added to C247 between comparison and addition
- Evaluator: SqlBinOp with op='||' already handled in C265/C266 evaluators
"""

import sys
import os

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C268_set_operations')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C267_common_table_expressions')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C266_subqueries')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C265_builtin_functions')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C264_window_functions')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C263_ctas')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C262_views')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C261_foreign_keys')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C260_check_constraints')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

from set_operations import (
    SetOpDB, SetOpParser, SetOperation, SetOpWithClauses,
    parse_set_op_sql, parse_set_op_sql_multi,
)
from mini_database import (
    ResultSet, ParseError, CompileError,
    SqlBinOp, SqlLiteral, SqlColumnRef,
    TokenType, Token, Lexer,
)


# Re-export SetOpDB as the main DB class for this challenge
StringConcatDB = SetOpDB
