"""
C251: SQL Triggers
Extends C250 (SQL Views) / C249 (Stored Procedures) / C247 (Mini Database Engine)

Adds SQL triggers to the database engine:
- CREATE TRIGGER name {BEFORE|AFTER} {INSERT|UPDATE|DELETE} ON table
  [FOR EACH ROW] [WHEN (condition)] BEGIN ... END
- DROP TRIGGER [IF EXISTS] name
- SHOW TRIGGERS [ON table]
- Row-level triggers with OLD and NEW pseudo-records
- BEFORE triggers can modify NEW values (INSERT/UPDATE)
- BEFORE triggers can cancel operations (via SIGNAL/RAISE)
- AFTER triggers execute after the operation completes
- Multiple triggers per table/event, ordered by creation time
- Trigger bodies support full SQL (INSERT, UPDATE, DELETE, SET, IF, etc.)
- Nested trigger detection (triggers firing triggers) with depth limit
- Trigger enable/disable (ALTER TRIGGER ... ENABLE/DISABLE)
- INSTEAD OF triggers on views
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set
from enum import Enum, auto

# Import C250 (which imports C249 -> C247)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C250_sql_views'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C249_stored_procedures'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from sql_views import (
    ViewDB, ViewLexer, ViewParser, ViewCatalog,
    CreateViewStmt, DropViewStmt, ShowViewsStmt, DescribeViewStmt,
    CheckOption, ViewDefinition,
)

from stored_procedures import (
    ProcDB, ProcLexer, ProcParser, ProcQueryCompiler, ProcExecutor,
    RoutineCatalog, ReturnSignal,
    CreateFunctionStmt, CreateProcedureStmt, DropFunctionStmt, DropProcedureStmt,
    CallStmt, ShowFunctionsStmt, ShowProceduresStmt,
    ParamDef, ParamMode,
    SetVarStmt, IfStmt, WhileStmt, DeclareVarStmt, ReturnStmt,
    SelectIntoStmt, DeclareHandlerStmt,
    LeaveStmt, LeaveSignal, IterateStmt, IterateSignal,
)

from mini_database import (
    MiniDB, ResultSet, DatabaseError,
    Lexer, Parser, Token, TokenType,
    parse_sql, parse_sql_multi,
    SelectStmt, SelectExpr, TableRef, JoinClause,
    InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    BeginStmt, CommitStmt, RollbackStmt, SavepointStmt,
    ShowTablesStmt, DescribeStmt, ExplainStmt,
    ColumnDef,
    SqlColumnRef, SqlLiteral, SqlBinOp, SqlComparison, SqlLogic,
    SqlIsNull, SqlFuncCall, SqlAggCall, SqlBetween, SqlInList,
    SqlCase, SqlStar,
    QueryCompiler, StorageEngine, CompileError, CatalogError,
    ParseError, KEYWORDS,
)

from query_executor import Row, eval_expr
from transaction_manager import IsolationLevel


# =============================================================================
# Trigger Timing and Event Enums
# =============================================================================

class TriggerTiming(Enum):
    BEFORE = auto()
    AFTER = auto()
    INSTEAD_OF = auto()


class TriggerEvent(Enum):
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()


# =============================================================================
# Trigger AST Nodes
# =============================================================================

@dataclass
class CreateTriggerStmt:
    name: str
    timing: TriggerTiming
    event: TriggerEvent
    table_name: str
    body: List[Any]  # list of SQL statements
    for_each_row: bool = True
    when_condition: Any = None  # optional SQL expression
    replace: bool = False
    update_columns: Optional[List[str]] = None  # UPDATE OF col1, col2


@dataclass
class DropTriggerStmt:
    name: str
    if_exists: bool = False


@dataclass
class ShowTriggersStmt:
    table_name: Optional[str] = None  # if specified, filter by table


@dataclass
class AlterTriggerStmt:
    name: str
    enable: bool = True  # True=ENABLE, False=DISABLE


@dataclass
class SignalStmt:
    """SIGNAL SQLSTATE 'xxxxx' SET MESSAGE_TEXT = '...'"""
    sqlstate: str = '45000'
    message: str = ''


# =============================================================================
# Trigger Definition
# =============================================================================

@dataclass
class TriggerDefinition:
    name: str
    timing: TriggerTiming
    event: TriggerEvent
    table_name: str
    body: List[Any]
    for_each_row: bool = True
    when_condition: Any = None
    enabled: bool = True
    update_columns: Optional[List[str]] = None
    creation_order: int = 0  # for ordering multiple triggers


# =============================================================================
# Trigger Catalog
# =============================================================================

class TriggerCatalog:
    """Manages trigger definitions."""

    def __init__(self):
        self._triggers: Dict[str, TriggerDefinition] = {}  # name -> definition
        self._table_triggers: Dict[str, List[str]] = {}  # table -> [trigger_names]
        self._creation_counter = 0

    def create_trigger(self, tdef: TriggerDefinition, replace: bool = False):
        name_lower = tdef.name.lower()
        if name_lower in self._triggers and not replace:
            raise DatabaseError(f"Trigger '{tdef.name}' already exists")

        self._creation_counter += 1
        tdef.creation_order = self._creation_counter

        # Remove old entry if replacing
        if name_lower in self._triggers:
            old = self._triggers[name_lower]
            tbl = old.table_name.lower()
            if tbl in self._table_triggers:
                self._table_triggers[tbl] = [
                    n for n in self._table_triggers[tbl] if n != name_lower
                ]

        self._triggers[name_lower] = tdef

        tbl = tdef.table_name.lower()
        if tbl not in self._table_triggers:
            self._table_triggers[tbl] = []
        self._table_triggers[tbl].append(name_lower)

    def drop_trigger(self, name: str, if_exists: bool = False):
        name_lower = name.lower()
        if name_lower not in self._triggers:
            if if_exists:
                return
            raise DatabaseError(f"Trigger '{name}' does not exist")

        tdef = self._triggers[name_lower]
        tbl = tdef.table_name.lower()
        if tbl in self._table_triggers:
            self._table_triggers[tbl] = [
                n for n in self._table_triggers[tbl] if n != name_lower
            ]
        del self._triggers[name_lower]

    def get_trigger(self, name: str) -> Optional[TriggerDefinition]:
        return self._triggers.get(name.lower())

    def get_triggers_for(self, table_name: str, timing: TriggerTiming,
                         event: TriggerEvent) -> List[TriggerDefinition]:
        """Get triggers for a table/timing/event, ordered by creation order."""
        tbl = table_name.lower()
        names = self._table_triggers.get(tbl, [])
        result = []
        for n in names:
            tdef = self._triggers[n]
            if tdef.timing == timing and tdef.event == event and tdef.enabled:
                result.append(tdef)
        result.sort(key=lambda t: t.creation_order)
        return result

    def list_triggers(self, table_name: Optional[str] = None) -> List[TriggerDefinition]:
        """List all triggers, optionally filtered by table."""
        if table_name:
            tbl = table_name.lower()
            names = self._table_triggers.get(tbl, [])
            return [self._triggers[n] for n in names]
        return list(self._triggers.values())

    def has_trigger(self, name: str) -> bool:
        return name.lower() in self._triggers

    def enable_trigger(self, name: str, enable: bool = True):
        name_lower = name.lower()
        if name_lower not in self._triggers:
            raise DatabaseError(f"Trigger '{name}' does not exist")
        self._triggers[name_lower].enabled = enable

    def get_triggers_for_table(self, table_name: str) -> List[str]:
        """Get all trigger names for a table."""
        return list(self._table_triggers.get(table_name.lower(), []))


# =============================================================================
# Trigger Lexer (extends ViewLexer)
# =============================================================================

TRIGGER_KEYWORDS = {
    'trigger': 'TRIGGER',
    'triggers': 'TRIGGERS',
    'before': 'BEFORE',
    'after': 'AFTER',
    'instead': 'INSTEAD',
    'each': 'EACH',
    'row': 'ROW',
    'signal': 'SIGNAL',
    'sqlstate': 'SQLSTATE',
    'message_text': 'MESSAGE_TEXT',
    # NOTE: 'new' and 'old' are intentionally NOT reclassified as ProcTokens.
    # They must remain IDENT so the standard SQL parser handles NEW.col and OLD.col
    # as table.column references (SqlColumnRef(table='NEW', column='col')).
    'enable': 'ENABLE',
    'disable': 'DISABLE',
    'then': 'THEN',
}

from stored_procedures import ProcToken


class TriggerLexer(ViewLexer):
    """Lexer extended with trigger keywords.
    Post-processes IDENT tokens to reclassify trigger keywords as ProcTokens."""

    def __init__(self, sql: str):
        super().__init__(sql)
        # Post-process: reclassify IDENT tokens that are trigger keywords
        for i, tok in enumerate(self.tokens):
            if hasattr(tok, 'type') and tok.type == TokenType.IDENT:
                lower = tok.value.lower() if isinstance(tok.value, str) else ''
                if lower in TRIGGER_KEYWORDS:
                    self.tokens[i] = ProcToken(TRIGGER_KEYWORDS[lower], tok.value, tok.pos)


# =============================================================================
# Trigger Parser (extends ViewParser)
# =============================================================================

class TriggerParser(ViewParser):
    """Parser extended with trigger statement parsing.
    Hooks into the _parse_create_extended / _parse_drop_extended / _parse_show_extended
    chain established by ProcParser/ViewParser."""

    def _peek_word(self) -> Optional[str]:
        """Peek at current token's value as lowercase string."""
        if self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            if hasattr(tok, 'value') and tok.value is not None:
                val = tok.value
                return val.lower() if isinstance(val, str) else None
        return None

    def _parse_statement(self):
        """Override to intercept ALTER TRIGGER."""
        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.ALTER:
            return self._parse_alter_trigger_maybe()
        return super()._parse_statement()

    def _parse_alter_trigger_maybe(self):
        """Parse ALTER TRIGGER ... ENABLE/DISABLE, or delegate to parent."""
        saved = self.pos
        self.advance()  # ALTER
        tt = self._peek_type()
        if tt == 'TRIGGER':
            self.advance()  # TRIGGER
            name_tok = self.advance()
            name = name_tok.value

            tt2 = self._peek_type()
            if tt2 == 'ENABLE':
                self.advance()
                return AlterTriggerStmt(name=name, enable=True)
            elif tt2 == 'DISABLE':
                self.advance()
                return AlterTriggerStmt(name=name, enable=False)
            else:
                raise ParseError(f"Expected ENABLE or DISABLE after ALTER TRIGGER {name}")
        # Not ALTER TRIGGER -- restore and let parent handle
        self.pos = saved
        return super()._parse_statement()

    def _parse_create_extended(self):
        """Override to handle CREATE [OR REPLACE] TRIGGER, delegate rest to parent.
        Called with CREATE not yet consumed."""
        saved = self.pos
        self.advance()  # CREATE

        # Check for OR REPLACE
        replace = False
        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.OR:
            save2 = self.pos
            self.advance()  # OR
            rtt = self._peek_type()
            if rtt == 'REPLACE':
                self.advance()  # REPLACE
                ntt = self._peek_type()
                if ntt == 'TRIGGER':
                    return self._parse_create_trigger(True)
            # Not TRIGGER -- restore fully and let parent handle
            self.pos = saved
            return super()._parse_create_extended()

        if tt == 'TRIGGER':
            return self._parse_create_trigger(False)

        # Not a trigger -- restore and let parent handle
        self.pos = saved
        return super()._parse_create_extended()

    def _parse_drop_extended(self):
        """Override to handle DROP TRIGGER, delegate rest to parent.
        Called with DROP not yet consumed."""
        saved = self.pos
        self.advance()  # DROP
        tt = self._peek_type()
        if tt == 'TRIGGER':
            self.advance()  # TRIGGER
            if_exists = False
            ptt = self._peek_type()
            if (isinstance(ptt, TokenType) and ptt == TokenType.IF) or ptt == 'IF':
                self.advance()  # IF
                self.expect(TokenType.EXISTS)
                if_exists = True
            name_tok = self.advance()
            return DropTriggerStmt(name=name_tok.value, if_exists=if_exists)
        # Not TRIGGER -- restore and let parent handle
        self.pos = saved
        return super()._parse_drop_extended()

    def _parse_show_extended(self):
        """Override to handle SHOW TRIGGERS, delegate rest to parent.
        Called with SHOW not yet consumed."""
        saved = self.pos
        self.advance()  # SHOW
        tt = self._peek_type()
        if tt == 'TRIGGERS':
            self.advance()  # TRIGGERS
            table_name = None
            ntt = self._peek_type()
            if isinstance(ntt, TokenType) and ntt == TokenType.ON:
                self.advance()  # ON
                table_name = self.advance().value
            return ShowTriggersStmt(table_name=table_name)
        # Not TRIGGERS -- restore and let parent handle
        self.pos = saved
        return super()._parse_show_extended()

    def _parse_create_trigger(self, replace: bool) -> CreateTriggerStmt:
        """Parse after CREATE [OR REPLACE] has been consumed.
        Current token should be TRIGGER."""
        self.advance()  # TRIGGER
        name_tok = self.advance()
        name = name_tok.value

        # Timing
        tt = self._peek_type()
        if tt == 'BEFORE':
            self.advance()
            timing = TriggerTiming.BEFORE
        elif tt == 'AFTER':
            self.advance()
            timing = TriggerTiming.AFTER
        elif tt == 'INSTEAD':
            self.advance()  # INSTEAD
            # Expect OF -- it might be TokenType.FULL or similar...
            # 'of' is not a keyword in our lexer, so handle it
            of_tok = self.advance()
            if of_tok.value.lower() != 'of':
                raise ParseError(f"Expected OF after INSTEAD, got '{of_tok.value}'")
            timing = TriggerTiming.INSTEAD_OF
        else:
            raise ParseError(f"Expected BEFORE, AFTER, or INSTEAD, got {tt}")

        # Event: INSERT, UPDATE, DELETE
        ett = self._peek_type()
        update_columns = None
        if isinstance(ett, TokenType) and ett == TokenType.INSERT:
            self.advance()
            event = TriggerEvent.INSERT
        elif isinstance(ett, TokenType) and ett == TokenType.UPDATE:
            self.advance()
            event = TriggerEvent.UPDATE
            # Optional OF col1, col2
            ntt = self._peek_type()
            if (isinstance(ntt, str) and ntt == 'OF') or self._peek_word() == 'of':
                self.advance()  # OF
                update_columns = [self.advance().value]
                while self._peek_type() == TokenType.COMMA:
                    self.advance()
                    update_columns.append(self.advance().value)
        elif isinstance(ett, TokenType) and ett == TokenType.DELETE:
            self.advance()
            event = TriggerEvent.DELETE
        else:
            raise ParseError(f"Expected INSERT, UPDATE, or DELETE, got {ett}")

        # ON table_name
        self.expect(TokenType.ON)
        table_tok = self.advance()
        table_name = table_tok.value

        # Optional FOR EACH ROW
        for_each_row = True
        ntt = self._peek_type()
        if ntt == 'FOR':
            self.advance()  # FOR
            self._expect_proc_word('EACH')
            self._expect_proc_word('ROW')

        # Optional WHEN (condition)
        when_condition = None
        ntt = self._peek_type()
        if isinstance(ntt, TokenType) and ntt == TokenType.WHEN:
            self.advance()  # WHEN
            self.expect(TokenType.LPAREN)
            when_condition = self._parse_trigger_expr()
            self.expect(TokenType.RPAREN)

        # BEGIN ... END block
        self.expect(TokenType.BEGIN)
        body = self._parse_trigger_body()
        self.expect(TokenType.END)

        return CreateTriggerStmt(
            name=name,
            timing=timing,
            event=event,
            table_name=table_name,
            body=body,
            for_each_row=for_each_row,
            when_condition=when_condition,
            replace=replace,
            update_columns=update_columns,
        )

    def _expect_proc_word(self, word: str):
        """Expect a ProcToken or ident with specific value."""
        tok = self.advance()
        if tok.value.lower() != word.lower():
            raise ParseError(f"Expected '{word}', got '{tok.value}'")

    def _parse_trigger_body(self) -> List[Any]:
        """Parse statements between BEGIN and END."""
        stmts = []
        while True:
            tt = self._peek_type()
            # END token means we're done
            if isinstance(tt, TokenType) and tt == TokenType.END:
                break
            if tt == TokenType.EOF:
                break

            pw = self._peek_word()
            if pw == 'signal':
                stmts.append(self._parse_signal())
            elif pw == 'set' or (isinstance(tt, TokenType) and tt == TokenType.SET):
                stmts.append(self._parse_set_in_trigger())
            elif pw == 'if' or (isinstance(tt, TokenType) and tt == TokenType.IF) or tt == 'IF':
                stmts.append(self._parse_trigger_if())
            elif pw == 'declare' or tt == 'DECLARE':
                stmts.append(self._parse_trigger_declare())
            else:
                # Regular SQL (INSERT, UPDATE, DELETE, SELECT)
                stmt = super()._parse_statement()
                stmts.append(stmt)
            # Consume optional semicolon
            if self.pos < len(self.tokens) and self.tokens[self.pos].type == TokenType.SEMICOLON:
                self.advance()
        return stmts

    def _parse_signal(self) -> SignalStmt:
        """Parse SIGNAL [SQLSTATE 'xxx'] [SET MESSAGE_TEXT = '...']"""
        self.advance()  # SIGNAL
        sqlstate = '45000'
        message = ''

        tt = self._peek_type()
        if tt == 'SQLSTATE':
            self.advance()
            str_tok = self.expect(TokenType.STRING)
            sqlstate = str_tok.value

        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.SET:
            self.advance()  # SET
            self._expect_proc_word('MESSAGE_TEXT')
            self.expect(TokenType.EQ)
            str_tok = self.expect(TokenType.STRING)
            message = str_tok.value

        return SignalStmt(sqlstate=sqlstate, message=message)

    def _parse_set_in_trigger(self):
        """Parse SET NEW.col = expr or SET var = expr in trigger body."""
        self.advance()  # SET

        # Check for NEW.col or OLD.col (NEW/OLD are IDENT tokens)
        pw = self._peek_word()
        if pw in ('new', 'old'):
            prefix = self.advance().value.lower()  # NEW or OLD
            self.expect(TokenType.DOT)
            col_tok = self.advance()
            col = col_tok.value
            self.expect(TokenType.EQ)
            expr = self._parse_trigger_expr()
            return TriggerSetStmt(prefix=prefix, column=col, expr=expr)
        else:
            # Regular SET var = expr
            name_tok = self.advance()
            name = name_tok.value
            self.expect(TokenType.EQ)
            expr = self._parse_trigger_expr()
            return SetVarStmt(name=name, expr=expr)

    def _parse_trigger_if(self):
        """Parse IF ... THEN ... [ELSEIF ... THEN ...] [ELSE ...] END IF"""
        self.advance()  # IF
        condition = self._parse_trigger_expr()
        self._expect_proc_word('THEN')

        then_body = self._parse_trigger_if_body()

        elseif_branches = []
        while self._peek_type() == 'ELSEIF' or self._peek_word() == 'elseif':
            self.advance()  # ELSEIF
            eif_cond = self._parse_trigger_expr()
            self._expect_proc_word('THEN')
            eif_body = self._parse_trigger_if_body()
            elseif_branches.append((eif_cond, eif_body))

        else_body = []
        tt = self._peek_type()
        if (isinstance(tt, TokenType) and tt == TokenType.ELSE) or tt == 'ELSE' or self._peek_word() == 'else':
            self.advance()  # ELSE
            else_body = self._parse_trigger_if_body()

        # END IF
        self.expect(TokenType.END)
        # IF might be TokenType.IF or ProcToken 'IF'
        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.IF:
            self.advance()
        elif tt == 'IF':
            self.advance()
        else:
            raise ParseError(f"Expected IF after END, got {tt}")

        return TriggerIfStmt(
            condition=condition,
            then_body=then_body,
            elseif_branches=elseif_branches,
            else_body=else_body
        )

    def _parse_trigger_if_body(self) -> List[Any]:
        """Parse statements inside an IF/ELSEIF/ELSE block (before ELSEIF/ELSE/END)."""
        stmts = []
        while True:
            tt = self._peek_type()
            pw = self._peek_word()
            # Stop at ELSEIF, ELSE, END
            if tt == 'ELSEIF' or pw == 'elseif':
                break
            if (isinstance(tt, TokenType) and tt == TokenType.ELSE) or tt == 'ELSE' or pw == 'else':
                break
            if isinstance(tt, TokenType) and tt == TokenType.END:
                break
            if tt == TokenType.EOF:
                break

            if pw == 'signal':
                stmts.append(self._parse_signal())
            elif pw == 'set' or (isinstance(tt, TokenType) and tt == TokenType.SET):
                stmts.append(self._parse_set_in_trigger())
            elif pw == 'if' or (isinstance(tt, TokenType) and tt == TokenType.IF) or tt == 'IF':
                stmts.append(self._parse_trigger_if())
            elif pw == 'declare' or tt == 'DECLARE':
                stmts.append(self._parse_trigger_declare())
            else:
                stmt = super()._parse_statement()
                stmts.append(stmt)
            # Consume optional semicolon
            if self.pos < len(self.tokens) and self.tokens[self.pos].type == TokenType.SEMICOLON:
                self.advance()
        return stmts

    def _parse_trigger_declare(self):
        """Parse DECLARE var_name type [DEFAULT value]"""
        self.advance()  # DECLARE
        name_tok = self.advance()
        name = name_tok.value
        type_tok = self.advance()  # type name (INT, VARCHAR, etc.)
        var_type = type_tok.value
        default = None
        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.DEFAULT:
            self.advance()
            default = self._parse_trigger_expr()
        return DeclareVarStmt(name=name, var_type=var_type, default=default)

    def _parse_trigger_expr(self):
        """Parse an expression in trigger context.
        Handles NEW.col, OLD.col references by transforming them to SqlColumnRef."""
        return self._parse_trigger_or_expr()

    def _parse_trigger_or_expr(self):
        left = self._parse_trigger_and_expr()
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.OR:
                self.advance()
                right = self._parse_trigger_and_expr()
                left = SqlLogic(op='or', operands=[left, right])
            else:
                break
        return left

    def _parse_trigger_and_expr(self):
        left = self._parse_trigger_not_expr()
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.AND:
                self.advance()
                right = self._parse_trigger_not_expr()
                left = SqlLogic(op='and', operands=[left, right])
            else:
                break
        return left

    def _parse_trigger_not_expr(self):
        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.NOT:
            self.advance()
            inner = self._parse_trigger_comparison()
            return SqlLogic(op='not', operands=[inner])
        return self._parse_trigger_comparison()

    def _parse_trigger_comparison(self):
        left = self._parse_trigger_addition()

        tt = self._peek_type()
        # IS [NOT] NULL
        if isinstance(tt, TokenType) and tt == TokenType.IS:
            self.advance()
            negated = False
            nt = self._peek_type()
            if isinstance(nt, TokenType) and nt == TokenType.NOT:
                self.advance()
                negated = True
            self.expect(TokenType.NULL)
            return SqlIsNull(expr=left, negated=negated)

        # Comparison operators
        cmp_ops = {
            TokenType.EQ: '=', TokenType.NE: '!=',
            TokenType.LT: '<', TokenType.GT: '>',
            TokenType.LE: '<=', TokenType.GE: '>=',
        }
        if isinstance(tt, TokenType) and tt in cmp_ops:
            op = cmp_ops[tt]
            self.advance()
            right = self._parse_trigger_addition()
            return SqlComparison(op=op, left=left, right=right)

        # BETWEEN
        if isinstance(tt, TokenType) and tt == TokenType.BETWEEN:
            self.advance()
            low = self._parse_trigger_addition()
            self.expect(TokenType.AND)
            high = self._parse_trigger_addition()
            return SqlBetween(expr=left, low=low, high=high)

        # IN (...)
        if isinstance(tt, TokenType) and tt == TokenType.IN:
            self.advance()
            self.expect(TokenType.LPAREN)
            vals = [self._parse_trigger_addition()]
            while self._peek_type() == TokenType.COMMA:
                self.advance()
                vals.append(self._parse_trigger_addition())
            self.expect(TokenType.RPAREN)
            return SqlInList(expr=left, values=vals)

        return left

    def _parse_trigger_addition(self):
        left = self._parse_trigger_multiplication()
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.PLUS:
                self.advance()
                right = self._parse_trigger_multiplication()
                left = SqlBinOp(op='+', left=left, right=right)
            elif isinstance(tt, TokenType) and tt == TokenType.MINUS:
                self.advance()
                right = self._parse_trigger_multiplication()
                left = SqlBinOp(op='-', left=left, right=right)
            elif isinstance(tt, str) and tt == 'CONCAT_OP':
                self.advance()
                right = self._parse_trigger_multiplication()
                left = SqlBinOp(op='||', left=left, right=right)
            else:
                break
        return left

    def _parse_trigger_multiplication(self):
        left = self._parse_trigger_unary()
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.STAR:
                self.advance()
                right = self._parse_trigger_unary()
                left = SqlBinOp(op='*', left=left, right=right)
            elif isinstance(tt, TokenType) and tt == TokenType.SLASH:
                self.advance()
                right = self._parse_trigger_unary()
                left = SqlBinOp(op='/', left=left, right=right)
            elif isinstance(tt, str) and tt == 'MOD':
                self.advance()
                right = self._parse_trigger_unary()
                left = SqlBinOp(op='%', left=left, right=right)
            else:
                break
        return left

    def _parse_trigger_unary(self):
        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.MINUS:
            self.advance()
            inner = self._parse_trigger_primary()
            return SqlBinOp(op='*', left=SqlLiteral(value=-1), right=inner)
        return self._parse_trigger_primary()

    def _parse_trigger_primary(self):
        tt = self._peek_type()
        tok = self.tokens[self.pos]

        # Number literal
        if isinstance(tt, TokenType) and tt == TokenType.NUMBER:
            self.advance()
            return SqlLiteral(value=tok.value)

        # String literal
        if isinstance(tt, TokenType) and tt == TokenType.STRING:
            self.advance()
            return SqlLiteral(value=tok.value)

        # NULL
        if isinstance(tt, TokenType) and tt == TokenType.NULL:
            self.advance()
            return SqlLiteral(value=None)

        # TRUE/FALSE
        if isinstance(tt, TokenType) and tt == TokenType.TRUE:
            self.advance()
            return SqlLiteral(value=True)
        if isinstance(tt, TokenType) and tt == TokenType.FALSE:
            self.advance()
            return SqlLiteral(value=False)

        # Parenthesized expression
        if isinstance(tt, TokenType) and tt == TokenType.LPAREN:
            self.advance()
            expr = self._parse_trigger_expr()
            self.expect(TokenType.RPAREN)
            return expr

        # CASE expression
        if isinstance(tt, TokenType) and tt == TokenType.CASE:
            return self._parse_trigger_case()

        # NEW.col / OLD.col (these are IDENT tokens since we don't reclassify them)
        if isinstance(tt, TokenType) and tt == TokenType.IDENT and tok.value.lower() in ('new', 'old'):
            prefix_tok = self.advance()
            if self._peek_type() == TokenType.DOT:
                self.advance()  # .
                col_tok = self.advance()
                return SqlColumnRef(table=prefix_tok.value, column=col_tok.value)
            # Just NEW/OLD without dot -- treat as identifier
            return SqlColumnRef(table=None, column=prefix_tok.value)

        # Function call or identifier
        if isinstance(tt, TokenType) and tt == TokenType.IDENT:
            name = tok.value
            self.advance()
            # Check for function call
            if self._peek_type() == TokenType.LPAREN:
                self.advance()  # (
                args = []
                if self._peek_type() != TokenType.RPAREN:
                    args.append(self._parse_trigger_expr())
                    while self._peek_type() == TokenType.COMMA:
                        self.advance()
                        args.append(self._parse_trigger_expr())
                self.expect(TokenType.RPAREN)
                return SqlFuncCall(func_name=name, args=args)
            # Check for table.column
            if self._peek_type() == TokenType.DOT:
                self.advance()
                col_tok = self.advance()
                return SqlColumnRef(table=name, column=col_tok.value)
            return SqlColumnRef(table=None, column=name)

        # Keyword-as-identifier (common: various proc/trigger keywords used as column names)
        if isinstance(tt, str) or isinstance(tt, TokenType):
            # Try consuming it as an identifier
            tok = self.advance()
            name = tok.value
            if isinstance(name, str):
                if self._peek_type() == TokenType.LPAREN:
                    self.advance()  # (
                    args = []
                    if self._peek_type() != TokenType.RPAREN:
                        args.append(self._parse_trigger_expr())
                        while self._peek_type() == TokenType.COMMA:
                            self.advance()
                            args.append(self._parse_trigger_expr())
                    self.expect(TokenType.RPAREN)
                    return SqlFuncCall(func_name=name, args=args)
                if self._peek_type() == TokenType.DOT:
                    self.advance()
                    col_tok = self.advance()
                    return SqlColumnRef(table=name, column=col_tok.value)
                return SqlColumnRef(table=None, column=name)

        raise ParseError(f"Unexpected token in trigger expression: {tt} ({tok.value!r})")

    def _parse_trigger_case(self):
        """Parse CASE WHEN ... THEN ... [ELSE ...] END"""
        self.advance()  # CASE
        whens = []
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.WHEN:
                self.advance()
                cond = self._parse_trigger_expr()
                # THEN might be TokenType.THEN or ProcToken 'THEN'
                ttt = self._peek_type()
                if isinstance(ttt, TokenType) and ttt == TokenType.THEN:
                    self.advance()
                elif ttt == 'THEN':
                    self.advance()
                else:
                    raise ParseError(f"Expected THEN, got {ttt}")
                result = self._parse_trigger_expr()
                whens.append((cond, result))
            else:
                break

        else_result = None
        tt = self._peek_type()
        if (isinstance(tt, TokenType) and tt == TokenType.ELSE) or tt == 'ELSE':
            self.advance()
            else_result = self._parse_trigger_expr()

        self.expect(TokenType.END)
        return SqlCase(whens=whens, else_result=else_result)


# =============================================================================
# Trigger-specific AST nodes
# =============================================================================

@dataclass
class TriggerSetStmt:
    """SET NEW.col = expr or SET OLD.col = expr (OLD is read-only at runtime)."""
    prefix: str  # 'new' or 'old'
    column: str
    expr: Any


@dataclass
class TriggerIfStmt:
    """IF/ELSEIF/ELSE in trigger body."""
    condition: Any
    then_body: List[Any]
    elseif_branches: List[Tuple[Any, List[Any]]]
    else_body: List[Any]


# =============================================================================
# Trigger Signal Exception
# =============================================================================

class TriggerSignal(Exception):
    """Raised by SIGNAL statement to abort the triggering operation."""
    def __init__(self, sqlstate: str = '45000', message: str = ''):
        self.sqlstate = sqlstate
        self.message = message
        super().__init__(message or f"SQLSTATE {sqlstate}")


# =============================================================================
# Trigger Executor
# =============================================================================

MAX_TRIGGER_DEPTH = 16


class TriggerExecutor:
    """Executes trigger bodies with access to OLD/NEW pseudo-records."""

    def __init__(self, db: 'TriggerDB'):
        self.db = db
        self._depth = 0

    def execute_trigger(self, tdef: TriggerDefinition,
                       old_row: Optional[Dict[str, Any]],
                       new_row: Optional[Dict[str, Any]],
                       local_vars: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Execute a trigger's body.

        Args:
            tdef: trigger definition
            old_row: the row before modification (UPDATE/DELETE), None for INSERT
            new_row: the row after modification (INSERT/UPDATE), None for DELETE
            local_vars: optional local variables (for DECLARE)

        Returns:
            Modified new_row (for BEFORE INSERT/UPDATE), or None
        """
        if self._depth >= MAX_TRIGGER_DEPTH:
            raise DatabaseError(
                f"Maximum trigger nesting depth ({MAX_TRIGGER_DEPTH}) exceeded"
            )

        self._depth += 1
        try:
            vars_ = local_vars or {}
            # Make OLD/NEW available as pseudo-dicts
            vars_['__old__'] = dict(old_row) if old_row else {}
            vars_['__new__'] = dict(new_row) if new_row else {}

            for stmt in tdef.body:
                self._exec_trigger_stmt(stmt, vars_)

            # Return potentially modified NEW row
            if new_row is not None:
                return vars_['__new__']
            return None
        finally:
            self._depth -= 1

    def _exec_trigger_stmt(self, stmt, vars_: Dict[str, Any]):
        """Execute a single statement in trigger context."""
        if isinstance(stmt, SignalStmt):
            raise TriggerSignal(stmt.sqlstate, stmt.message)

        if isinstance(stmt, TriggerSetStmt):
            value = self._eval_trigger_expr(stmt.expr, vars_)
            if stmt.prefix == 'new':
                vars_['__new__'][stmt.column] = value
            elif stmt.prefix == 'old':
                raise DatabaseError("Cannot modify OLD values in trigger")
            else:
                raise DatabaseError(f"Unknown prefix '{stmt.prefix}' in SET")

        elif isinstance(stmt, SetVarStmt):
            value = self._eval_trigger_expr(stmt.expr, vars_)
            vars_[stmt.name] = value

        elif isinstance(stmt, DeclareVarStmt):
            if stmt.default is not None:
                vars_[stmt.name] = self._eval_trigger_expr(stmt.default, vars_)
            else:
                vars_[stmt.name] = None

        elif isinstance(stmt, TriggerIfStmt):
            self._exec_trigger_if(stmt, vars_)

        elif isinstance(stmt, InsertStmt):
            # Substitute NEW/OLD references in the statement
            resolved = self._resolve_refs_in_insert(stmt, vars_)
            self.db._execute_with_triggers(resolved)

        elif isinstance(stmt, UpdateStmt):
            resolved = self._resolve_refs_in_update(stmt, vars_)
            self.db._execute_with_triggers(resolved)

        elif isinstance(stmt, DeleteStmt):
            resolved = self._resolve_refs_in_delete(stmt, vars_)
            self.db._execute_with_triggers(resolved)

        elif isinstance(stmt, SelectStmt):
            # Execute SELECT for side effects (e.g., INTO)
            self.db._execute_with_triggers(stmt)

        elif isinstance(stmt, SelectIntoStmt):
            self._exec_select_into(stmt, vars_)

        else:
            # Unknown -- try as regular SQL
            self.db._execute_with_triggers(stmt)

    def _exec_trigger_if(self, stmt: TriggerIfStmt, vars_: Dict[str, Any]):
        """Execute IF/ELSEIF/ELSE in trigger body."""
        cond_val = self._eval_trigger_expr(stmt.condition, vars_)
        if cond_val:
            for s in stmt.then_body:
                self._exec_trigger_stmt(s, vars_)
            return

        for cond, body in stmt.elseif_branches:
            if self._eval_trigger_expr(cond, vars_):
                for s in body:
                    self._exec_trigger_stmt(s, vars_)
                return

        for s in stmt.else_body:
            self._exec_trigger_stmt(s, vars_)

    def _eval_trigger_expr(self, expr, vars_: Dict[str, Any]) -> Any:
        """Evaluate an expression in trigger context, resolving OLD/NEW references."""
        if expr is None:
            return None

        if isinstance(expr, SqlLiteral):
            return expr.value

        if isinstance(expr, SqlColumnRef):
            # Check for NEW.col / OLD.col
            if expr.table and expr.table.lower() == 'new':
                return vars_.get('__new__', {}).get(expr.column)
            if expr.table and expr.table.lower() == 'old':
                return vars_.get('__old__', {}).get(expr.column)
            # Check local variables
            if expr.column in vars_ and expr.column not in ('__old__', '__new__'):
                return vars_[expr.column]
            # Plain column reference -- look in NEW then OLD
            new_row = vars_.get('__new__', {})
            if expr.column in new_row:
                return new_row[expr.column]
            old_row = vars_.get('__old__', {})
            if expr.column in old_row:
                return old_row[expr.column]
            return None

        if isinstance(expr, SqlBinOp):
            left = self._eval_trigger_expr(expr.left, vars_)
            right = self._eval_trigger_expr(expr.right, vars_)
            if left is None or right is None:
                return None
            if expr.op == '+': return left + right
            if expr.op == '-': return left - right
            if expr.op == '*': return left * right
            if expr.op == '/':
                if right == 0:
                    raise DatabaseError("Division by zero")
                return left / right
            if expr.op == '%': return left % right
            if expr.op == '||': return str(left) + str(right)

        if isinstance(expr, SqlComparison):
            left = self._eval_trigger_expr(expr.left, vars_)
            right = self._eval_trigger_expr(expr.right, vars_)
            if expr.op == '=': return left == right
            if expr.op == '!=': return left != right
            if expr.op == '<>': return left != right
            if expr.op == '<': return left < right
            if expr.op == '>': return left > right
            if expr.op == '<=': return left <= right
            if expr.op == '>=': return left >= right

        if isinstance(expr, SqlLogic):
            if expr.op == 'and':
                return all(self._eval_trigger_expr(o, vars_) for o in expr.operands)
            if expr.op == 'or':
                return any(self._eval_trigger_expr(o, vars_) for o in expr.operands)
            if expr.op == 'not':
                return not self._eval_trigger_expr(expr.operands[0], vars_)

        if isinstance(expr, SqlIsNull):
            val = self._eval_trigger_expr(expr.expr, vars_)
            is_null = val is None
            return not is_null if expr.negated else is_null

        if isinstance(expr, SqlFuncCall):
            args = [self._eval_trigger_expr(a, vars_) for a in expr.args]
            fname = expr.func_name.lower()
            if fname == 'upper' and args:
                return str(args[0]).upper() if args[0] is not None else None
            if fname == 'lower' and args:
                return str(args[0]).lower() if args[0] is not None else None
            if fname == 'length' and args:
                return len(str(args[0])) if args[0] is not None else None
            if fname == 'abs' and args:
                return abs(args[0]) if args[0] is not None else None
            if fname == 'coalesce':
                for a in args:
                    if a is not None:
                        return a
                return None
            if fname == 'concat':
                return ''.join(str(a) for a in args if a is not None)
            # Fallback: try as UDF
            if self.db.routines and fname in self.db.routines.functions:
                executor = ProcExecutor(self.db)
                return executor.call_function(fname, args)
            return None

        if isinstance(expr, SqlCase):
            for when_cond, when_result in expr.whens:
                if self._eval_trigger_expr(when_cond, vars_):
                    return self._eval_trigger_expr(when_result, vars_)
            if expr.else_result:
                return self._eval_trigger_expr(expr.else_result, vars_)
            return None

        if isinstance(expr, SqlBetween):
            val = self._eval_trigger_expr(expr.expr, vars_)
            low = self._eval_trigger_expr(expr.low, vars_)
            high = self._eval_trigger_expr(expr.high, vars_)
            return low <= val <= high

        if isinstance(expr, SqlInList):
            val = self._eval_trigger_expr(expr.expr, vars_)
            vals = [self._eval_trigger_expr(v, vars_) for v in expr.values]
            return val in vals

        return expr

    def check_when_condition(self, tdef: TriggerDefinition,
                            old_row: Optional[Dict[str, Any]],
                            new_row: Optional[Dict[str, Any]]) -> bool:
        """Evaluate WHEN condition. Returns True if trigger should fire."""
        if tdef.when_condition is None:
            return True

        vars_ = {
            '__old__': dict(old_row) if old_row else {},
            '__new__': dict(new_row) if new_row else {},
        }
        result = self._eval_trigger_expr(tdef.when_condition, vars_)
        return bool(result)

    def check_update_columns(self, tdef: TriggerDefinition,
                            updated_columns: Optional[Set[str]]) -> bool:
        """Check if trigger should fire for UPDATE OF specific columns."""
        if tdef.update_columns is None:
            return True
        if updated_columns is None:
            return True
        target_cols = {c.lower() for c in tdef.update_columns}
        actual_cols = {c.lower() for c in updated_columns}
        return bool(target_cols & actual_cols)

    # -- Statement reference resolution --

    def _resolve_refs_in_insert(self, stmt: InsertStmt, vars_: Dict[str, Any]) -> InsertStmt:
        """Resolve NEW/OLD references in INSERT values."""
        new_values_list = []
        for values in stmt.values_list:
            new_values = []
            for v in values:
                new_values.append(self._resolve_expr_refs(v, vars_))
            new_values_list.append(new_values)
        return InsertStmt(
            table_name=stmt.table_name,
            columns=stmt.columns,
            values_list=new_values_list
        )

    def _resolve_refs_in_update(self, stmt: UpdateStmt, vars_: Dict[str, Any]) -> UpdateStmt:
        """Resolve NEW/OLD references in UPDATE assignments and WHERE."""
        new_assignments = []
        for col, expr in stmt.assignments:
            new_assignments.append((col, self._resolve_expr_refs(expr, vars_)))
        new_where = self._resolve_expr_refs(stmt.where, vars_) if stmt.where else None
        return UpdateStmt(
            table_name=stmt.table_name,
            assignments=new_assignments,
            where=new_where
        )

    def _resolve_refs_in_delete(self, stmt: DeleteStmt, vars_: Dict[str, Any]) -> DeleteStmt:
        """Resolve NEW/OLD references in DELETE WHERE."""
        new_where = self._resolve_expr_refs(stmt.where, vars_) if stmt.where else None
        return DeleteStmt(
            table_name=stmt.table_name,
            where=new_where
        )

    def _resolve_expr_refs(self, expr, vars_: Dict[str, Any]):
        """Replace NEW.col / OLD.col / local var references with literal values."""
        if expr is None:
            return None

        if isinstance(expr, SqlColumnRef):
            if expr.table and expr.table.lower() == 'new':
                val = vars_.get('__new__', {}).get(expr.column)
                return SqlLiteral(value=val)
            if expr.table and expr.table.lower() == 'old':
                val = vars_.get('__old__', {}).get(expr.column)
                return SqlLiteral(value=val)
            # Local variable
            if expr.column in vars_ and expr.column not in ('__old__', '__new__'):
                return SqlLiteral(value=vars_[expr.column])
            return expr

        if isinstance(expr, SqlBinOp):
            return SqlBinOp(
                op=expr.op,
                left=self._resolve_expr_refs(expr.left, vars_),
                right=self._resolve_expr_refs(expr.right, vars_)
            )

        if isinstance(expr, SqlComparison):
            return SqlComparison(
                op=expr.op,
                left=self._resolve_expr_refs(expr.left, vars_),
                right=self._resolve_expr_refs(expr.right, vars_)
            )

        if isinstance(expr, SqlLogic):
            return SqlLogic(
                op=expr.op,
                operands=[self._resolve_expr_refs(o, vars_) for o in expr.operands]
            )

        if isinstance(expr, SqlIsNull):
            return SqlIsNull(
                expr=self._resolve_expr_refs(expr.expr, vars_),
                negated=expr.negated
            )

        if isinstance(expr, SqlFuncCall):
            return SqlFuncCall(
                func_name=expr.func_name,
                args=[self._resolve_expr_refs(a, vars_) for a in expr.args]
            )

        return expr

    def _exec_select_into(self, stmt: SelectIntoStmt, vars_: Dict[str, Any]):
        """Execute SELECT ... INTO in trigger body."""
        # Build a regular SELECT and execute it
        select = SelectStmt(
            columns=stmt.columns if hasattr(stmt, 'columns') else stmt.select.columns,
            from_table=stmt.from_table if hasattr(stmt, 'from_table') else stmt.select.from_table,
            joins=stmt.joins if hasattr(stmt, 'joins') else getattr(stmt.select, 'joins', []),
            where=stmt.where if hasattr(stmt, 'where') else stmt.select.where,
            group_by=[],
            having=None,
            order_by=None,
            limit=1,
            offset=None,
            distinct=False,
        )
        result = self.db._execute_with_triggers(select)
        if result.rows:
            for i, var_name in enumerate(stmt.into_vars):
                if i < len(result.rows[0]):
                    vars_[var_name] = result.rows[0][i]


# =============================================================================
# TriggerDB -- Main Database with Trigger Support
# =============================================================================

class TriggerDB(ViewDB):
    """ViewDB extended with SQL triggers."""

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)
        self.trigger_catalog = TriggerCatalog()
        self.trigger_executor = TriggerExecutor(self)
        self._in_trigger = False  # flag to prevent double-firing

    def execute(self, sql: str) -> 'ResultSet':
        stmts = self._parse_triggers(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_trigger_stmt(stmt))
        return results[-1] if results else ResultSet(columns=[], rows=[], message="OK")

    def execute_many(self, sql: str) -> List['ResultSet']:
        stmts = self._parse_triggers(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_trigger_stmt(stmt))
        return results

    def _parse_triggers(self, sql: str) -> List[Any]:
        lexer = TriggerLexer(sql)
        parser = TriggerParser(lexer.tokens)
        stmts = []
        while parser._peek_type() != TokenType.EOF:
            stmts.append(parser._parse_statement())
            parser.match(TokenType.SEMICOLON)
        return stmts

    def _execute_trigger_stmt(self, stmt) -> 'ResultSet':
        # Trigger-specific DDL
        if isinstance(stmt, CreateTriggerStmt):
            return self._exec_create_trigger(stmt)
        if isinstance(stmt, DropTriggerStmt):
            return self._exec_drop_trigger(stmt)
        if isinstance(stmt, ShowTriggersStmt):
            return self._exec_show_triggers(stmt)
        if isinstance(stmt, AlterTriggerStmt):
            return self._exec_alter_trigger(stmt)

        # DML with trigger support
        if isinstance(stmt, InsertStmt):
            # Check if target is a view first
            if self.view_catalog.has_view(stmt.table_name):
                # Check for INSTEAD OF trigger on view
                instead_triggers = self.trigger_catalog.get_triggers_for(
                    stmt.table_name, TriggerTiming.INSTEAD_OF, TriggerEvent.INSERT
                )
                if instead_triggers:
                    return self._exec_instead_of_insert(stmt, instead_triggers)
                return self._exec_insert_through_view(stmt)
            return self._exec_insert_with_triggers(stmt)

        if isinstance(stmt, UpdateStmt):
            if self.view_catalog.has_view(stmt.table_name):
                instead_triggers = self.trigger_catalog.get_triggers_for(
                    stmt.table_name, TriggerTiming.INSTEAD_OF, TriggerEvent.UPDATE
                )
                if instead_triggers:
                    return self._exec_instead_of_update(stmt, instead_triggers)
                return self._exec_update_through_view(stmt)
            return self._exec_update_with_triggers(stmt)

        if isinstance(stmt, DeleteStmt):
            if self.view_catalog.has_view(stmt.table_name):
                instead_triggers = self.trigger_catalog.get_triggers_for(
                    stmt.table_name, TriggerTiming.INSTEAD_OF, TriggerEvent.DELETE
                )
                if instead_triggers:
                    return self._exec_instead_of_delete(stmt, instead_triggers)
                return self._exec_delete_through_view(stmt)
            return self._exec_delete_with_triggers(stmt)

        # Drop table should also drop associated triggers
        if isinstance(stmt, DropTableStmt):
            return self._exec_drop_table_with_triggers(stmt)

        # Everything else delegates to parent (view expansion, etc.)
        return self._execute_view_stmt(stmt)

    def _execute_stmt(self, stmt) -> 'ResultSet':
        """Override MiniDB._execute_stmt to route DML through trigger-aware paths.
        This ensures triggers fire even when DML comes from stored procedures."""
        if isinstance(stmt, InsertStmt):
            if self.view_catalog.has_view(stmt.table_name):
                instead_triggers = self.trigger_catalog.get_triggers_for(
                    stmt.table_name, TriggerTiming.INSTEAD_OF, TriggerEvent.INSERT
                )
                if instead_triggers:
                    return self._exec_instead_of_insert(stmt, instead_triggers)
                return self._exec_insert_through_view(stmt)
            return self._exec_insert_with_triggers(stmt)

        if isinstance(stmt, UpdateStmt):
            if self.view_catalog.has_view(stmt.table_name):
                instead_triggers = self.trigger_catalog.get_triggers_for(
                    stmt.table_name, TriggerTiming.INSTEAD_OF, TriggerEvent.UPDATE
                )
                if instead_triggers:
                    return self._exec_instead_of_update(stmt, instead_triggers)
                return self._exec_update_through_view(stmt)
            return self._exec_update_with_triggers(stmt)

        if isinstance(stmt, DeleteStmt):
            if self.view_catalog.has_view(stmt.table_name):
                instead_triggers = self.trigger_catalog.get_triggers_for(
                    stmt.table_name, TriggerTiming.INSTEAD_OF, TriggerEvent.DELETE
                )
                if instead_triggers:
                    return self._exec_instead_of_delete(stmt, instead_triggers)
                return self._exec_delete_through_view(stmt)
            return self._exec_delete_with_triggers(stmt)

        if isinstance(stmt, DropTableStmt):
            return self._exec_drop_table_with_triggers(stmt)

        return super()._execute_stmt(stmt)

    def _execute_with_triggers(self, stmt) -> 'ResultSet':
        """Execute a statement (used by trigger bodies). Dispatches to trigger-aware exec."""
        return self._execute_trigger_stmt(stmt)

    # -- Trigger DDL --

    def _exec_create_trigger(self, stmt: CreateTriggerStmt) -> ResultSet:
        # Validate table/view exists
        table_lower = stmt.table_name.lower()
        is_view = self.view_catalog.has_view(table_lower)
        is_table = self.storage.catalog.has_table(table_lower)

        if not is_view and not is_table:
            raise DatabaseError(f"Table or view '{stmt.table_name}' does not exist")

        # INSTEAD OF only on views
        if stmt.timing == TriggerTiming.INSTEAD_OF and not is_view:
            raise DatabaseError("INSTEAD OF triggers can only be created on views")

        # BEFORE/AFTER not on views (unless we allow it for tables underlying views)
        if stmt.timing in (TriggerTiming.BEFORE, TriggerTiming.AFTER) and is_view and not is_table:
            raise DatabaseError(
                f"BEFORE/AFTER triggers cannot be created on views. Use INSTEAD OF."
            )

        tdef = TriggerDefinition(
            name=stmt.name,
            timing=stmt.timing,
            event=stmt.event,
            table_name=stmt.table_name,
            body=stmt.body,
            for_each_row=stmt.for_each_row,
            when_condition=stmt.when_condition,
            update_columns=stmt.update_columns,
        )
        self.trigger_catalog.create_trigger(tdef, replace=stmt.replace)
        return ResultSet(columns=[], rows=[], message=f"CREATE TRIGGER {stmt.name}")

    def _exec_drop_trigger(self, stmt: DropTriggerStmt) -> ResultSet:
        self.trigger_catalog.drop_trigger(stmt.name, if_exists=stmt.if_exists)
        return ResultSet(columns=[], rows=[], message=f"DROP TRIGGER {stmt.name}")

    def _exec_show_triggers(self, stmt: ShowTriggersStmt) -> ResultSet:
        triggers = self.trigger_catalog.list_triggers(stmt.table_name)
        rows = []
        for t in triggers:
            rows.append([
                t.name,
                t.timing.name,
                t.event.name,
                t.table_name,
                'ENABLED' if t.enabled else 'DISABLED',
            ])
        return ResultSet(
            columns=['trigger_name', 'timing', 'event', 'table_name', 'status'],
            rows=rows
        )

    def _exec_alter_trigger(self, stmt: AlterTriggerStmt) -> ResultSet:
        self.trigger_catalog.enable_trigger(stmt.name, stmt.enable)
        action = 'ENABLE' if stmt.enable else 'DISABLE'
        return ResultSet(columns=[], rows=[], message=f"ALTER TRIGGER {stmt.name} {action}")

    def _exec_drop_table_with_triggers(self, stmt: DropTableStmt) -> ResultSet:
        """Drop table and any associated triggers."""
        # Check view dependencies first
        dependents = self.view_catalog.get_dependents(stmt.table_name)
        if dependents:
            raise DatabaseError(
                f"Cannot drop table '{stmt.table_name}': referenced by view(s) {', '.join(dependents)}"
            )

        # Drop triggers on this table
        trigger_names = self.trigger_catalog.get_triggers_for_table(stmt.table_name)
        for tname in list(trigger_names):
            self.trigger_catalog.drop_trigger(tname, if_exists=True)

        # Drop the table
        return super()._execute_view_stmt(stmt)

    # -- INSERT with triggers --

    def _exec_insert_with_triggers(self, stmt: InsertStmt) -> ResultSet:
        table_name = stmt.table_name
        schema = self.storage.catalog.get_table(table_name)
        col_names = stmt.columns or schema.column_names()

        before_triggers = self.trigger_catalog.get_triggers_for(
            table_name, TriggerTiming.BEFORE, TriggerEvent.INSERT
        )
        after_triggers = self.trigger_catalog.get_triggers_for(
            table_name, TriggerTiming.AFTER, TriggerEvent.INSERT
        )

        # If no triggers, fast path
        if not before_triggers and not after_triggers:
            return super()._execute_view_stmt(stmt)

        txn_id = self._get_txn()
        try:
            count = 0
            inserted_rows = []

            for values in stmt.values_list:
                # Build NEW row
                new_row = {}
                for i, col in enumerate(col_names):
                    if i < len(values):
                        new_row[col] = self._eval_sql_value(values[i])
                    else:
                        new_row[col] = None

                # BEFORE triggers
                for tdef in before_triggers:
                    if self.trigger_executor.check_when_condition(tdef, None, new_row):
                        try:
                            modified = self.trigger_executor.execute_trigger(
                                tdef, old_row=None, new_row=new_row
                            )
                            if modified is not None:
                                new_row = modified
                        except TriggerSignal as e:
                            self._auto_abort(txn_id)
                            raise DatabaseError(
                                f"Trigger '{tdef.name}': {e.message}" if e.message
                                else f"Trigger '{tdef.name}' raised SQLSTATE {e.sqlstate}"
                            )

                # Perform the insert
                self.storage.insert_row(txn_id, table_name, new_row)
                count += 1
                inserted_rows.append(new_row)

            # Commit BEFORE firing AFTER triggers (correct semantics)
            self._auto_commit(txn_id)

            # AFTER triggers (data is committed, nested DML will see it)
            for new_row in inserted_rows:
                for tdef in after_triggers:
                    if self.trigger_executor.check_when_condition(tdef, None, new_row):
                        try:
                            self.trigger_executor.execute_trigger(
                                tdef, old_row=None, new_row=new_row
                            )
                        except TriggerSignal as e:
                            raise DatabaseError(
                                f"Trigger '{tdef.name}': {e.message}" if e.message
                                else f"Trigger '{tdef.name}' raised SQLSTATE {e.sqlstate}"
                            )

            return ResultSet(columns=[], rows=[], message=f"INSERT {count}",
                           rows_affected=count)
        except DatabaseError:
            raise
        except Exception:
            self._auto_abort(txn_id)
            raise

    # -- UPDATE with triggers --

    def _exec_update_with_triggers(self, stmt: UpdateStmt) -> ResultSet:
        table_name = stmt.table_name

        before_triggers = self.trigger_catalog.get_triggers_for(
            table_name, TriggerTiming.BEFORE, TriggerEvent.UPDATE
        )
        after_triggers = self.trigger_catalog.get_triggers_for(
            table_name, TriggerTiming.AFTER, TriggerEvent.UPDATE
        )

        # Check UPDATE OF column filtering
        updated_cols = {col for col, _ in stmt.assignments} if stmt.assignments else None

        # Filter triggers by UPDATE OF columns
        before_triggers = [
            t for t in before_triggers
            if self.trigger_executor.check_update_columns(t, updated_cols)
        ]
        after_triggers = [
            t for t in after_triggers
            if self.trigger_executor.check_update_columns(t, updated_cols)
        ]

        # If no triggers, fast path
        if not before_triggers and not after_triggers:
            return super()._execute_view_stmt(stmt)

        txn_id = self._get_txn()
        try:
            schema = self.storage.catalog.get_table(table_name)
            all_rows = self.storage.scan_table(txn_id, table_name)
            count = 0
            after_data = []  # (old_row, new_row) pairs for AFTER triggers

            for rowid, row_data in all_rows:
                # Check WHERE
                if stmt.where is not None:
                    qe_row = Row(row_data)
                    qe_pred = self.compiler._sql_to_qe_expr(stmt.where)
                    if not eval_expr(qe_pred, qe_row):
                        continue

                old_row = dict(row_data)

                # Build NEW row
                new_row = dict(row_data)
                for col, val_expr in stmt.assignments:
                    if isinstance(val_expr, SqlLiteral):
                        new_row[col] = val_expr.value
                    else:
                        qe_row = Row(row_data)
                        qe_expr = self.compiler._sql_to_qe_expr(val_expr)
                        new_row[col] = eval_expr(qe_expr, qe_row)

                # BEFORE triggers
                for tdef in before_triggers:
                    if self.trigger_executor.check_when_condition(tdef, old_row, new_row):
                        try:
                            modified = self.trigger_executor.execute_trigger(
                                tdef, old_row=old_row, new_row=new_row
                            )
                            if modified is not None:
                                new_row = modified
                        except TriggerSignal as e:
                            self._auto_abort(txn_id)
                            raise DatabaseError(
                                f"Trigger '{tdef.name}': {e.message}" if e.message
                                else f"Trigger '{tdef.name}' raised SQLSTATE {e.sqlstate}"
                            )

                # Compute actual updates from (potentially modified) new_row
                updates = {}
                for col in new_row:
                    if col in old_row and new_row[col] != old_row[col]:
                        updates[col] = new_row[col]
                    elif col not in old_row:
                        updates[col] = new_row[col]
                # If BEFORE trigger modified NEW, we need to apply ALL assigned cols
                # plus any trigger modifications
                if not updates:
                    # Even if values didn't change, we need to apply the update
                    for col, _ in stmt.assignments:
                        updates[col] = new_row[col]

                if updates:
                    self.storage.update_row(txn_id, table_name, rowid, updates)
                count += 1
                after_data.append((old_row, new_row))

            # Commit BEFORE firing AFTER triggers
            self._auto_commit(txn_id)

            # AFTER triggers (data committed, nested DML sees it)
            for old_row, new_row in after_data:
                for tdef in after_triggers:
                    if self.trigger_executor.check_when_condition(tdef, old_row, new_row):
                        try:
                            self.trigger_executor.execute_trigger(
                                tdef, old_row=old_row, new_row=new_row
                            )
                        except TriggerSignal as e:
                            raise DatabaseError(
                                f"Trigger '{tdef.name}': {e.message}" if e.message
                                else f"Trigger '{tdef.name}' raised SQLSTATE {e.sqlstate}"
                            )

            return ResultSet(columns=[], rows=[], message=f"UPDATE {count}",
                           rows_affected=count)
        except DatabaseError:
            raise
        except Exception:
            self._auto_abort(txn_id)
            raise

    # -- DELETE with triggers --

    def _exec_delete_with_triggers(self, stmt: DeleteStmt) -> ResultSet:
        table_name = stmt.table_name

        before_triggers = self.trigger_catalog.get_triggers_for(
            table_name, TriggerTiming.BEFORE, TriggerEvent.DELETE
        )
        after_triggers = self.trigger_catalog.get_triggers_for(
            table_name, TriggerTiming.AFTER, TriggerEvent.DELETE
        )

        # If no triggers, fast path
        if not before_triggers and not after_triggers:
            return super()._execute_view_stmt(stmt)

        txn_id = self._get_txn()
        try:
            all_rows = self.storage.scan_table(txn_id, table_name)
            count = 0
            deleted_rows = []

            for rowid, row_data in all_rows:
                if stmt.where is not None:
                    qe_row = Row(row_data)
                    qe_pred = self.compiler._sql_to_qe_expr(stmt.where)
                    if not eval_expr(qe_pred, qe_row):
                        continue

                old_row = dict(row_data)

                # BEFORE triggers
                cancelled = False
                for tdef in before_triggers:
                    if self.trigger_executor.check_when_condition(tdef, old_row, None):
                        try:
                            self.trigger_executor.execute_trigger(
                                tdef, old_row=old_row, new_row=None
                            )
                        except TriggerSignal as e:
                            self._auto_abort(txn_id)
                            raise DatabaseError(
                                f"Trigger '{tdef.name}': {e.message}" if e.message
                                else f"Trigger '{tdef.name}' raised SQLSTATE {e.sqlstate}"
                            )

                self.storage.delete_row(txn_id, table_name, rowid)
                count += 1
                deleted_rows.append(old_row)

            # Commit BEFORE firing AFTER triggers
            self._auto_commit(txn_id)

            # AFTER triggers (data committed, nested DML sees it)
            for old_row in deleted_rows:
                for tdef in after_triggers:
                    if self.trigger_executor.check_when_condition(tdef, old_row, None):
                        try:
                            self.trigger_executor.execute_trigger(
                                tdef, old_row=old_row, new_row=None
                            )
                        except TriggerSignal as e:
                            raise DatabaseError(
                                f"Trigger '{tdef.name}': {e.message}" if e.message
                                else f"Trigger '{tdef.name}' raised SQLSTATE {e.sqlstate}"
                            )

            return ResultSet(columns=[], rows=[], message=f"DELETE {count}",
                           rows_affected=count)
        except DatabaseError:
            raise
        except Exception:
            self._auto_abort(txn_id)
            raise

    # -- View-through DML: route through trigger-aware execution --

    def _exec_insert_through_view(self, stmt: InsertStmt) -> ResultSet:
        """Override: after view resolves to base table, fire triggers on base table."""
        vdef, base_table = self._get_updatable_view(stmt.table_name)
        col_map, final_base, top_vdef = self._get_full_column_mapping(stmt.table_name)

        new_columns = None
        if stmt.columns:
            new_columns = [col_map.get(c, c) for c in stmt.columns]

        new_stmt = InsertStmt(
            table_name=final_base,
            columns=new_columns,
            values_list=stmt.values_list
        )
        # Route through trigger-aware INSERT
        result = self._exec_insert_with_triggers(new_stmt)

        if top_vdef.check_option != CheckOption.NONE:
            self._check_option_after_insert(top_vdef, stmt.values_list, new_columns, final_base)

        return result

    def _exec_update_through_view(self, stmt: UpdateStmt) -> ResultSet:
        """Override: after view resolves to base table, fire triggers on base table."""
        vdef, base_table = self._get_updatable_view(stmt.table_name)
        col_map, final_base, top_vdef = self._get_full_column_mapping(stmt.table_name)

        new_assignments = []
        for col, expr in stmt.assignments:
            mapped_col = col_map.get(col, col)
            new_expr = self._remap_col_refs(expr, col_map)
            new_assignments.append((mapped_col, new_expr))

        view_where = self._get_effective_where(stmt.table_name)
        combined_where = self._remap_col_refs(stmt.where, col_map) if stmt.where else None
        if view_where and combined_where:
            combined_where = SqlLogic(op='and', operands=[view_where, combined_where])
        elif view_where:
            combined_where = view_where

        new_stmt = UpdateStmt(
            table_name=final_base,
            assignments=new_assignments,
            where=combined_where
        )
        # Route through trigger-aware UPDATE
        result = self._exec_update_with_triggers(new_stmt)

        if top_vdef.check_option != CheckOption.NONE and view_where:
            self._check_option_after_update(top_vdef, final_base, combined_where)

        return result

    def _exec_delete_through_view(self, stmt: DeleteStmt) -> ResultSet:
        """Override: after view resolves to base table, fire triggers on base table."""
        vdef, base_table = self._get_updatable_view(stmt.table_name)
        col_map, final_base, top_vdef = self._get_full_column_mapping(stmt.table_name)

        view_where = self._get_effective_where(stmt.table_name)
        combined_where = self._remap_col_refs(stmt.where, col_map) if stmt.where else None
        if view_where and combined_where:
            combined_where = SqlLogic(op='and', operands=[view_where, combined_where])
        elif view_where:
            combined_where = view_where

        new_stmt = DeleteStmt(
            table_name=final_base,
            where=combined_where
        )
        # Route through trigger-aware DELETE
        return self._exec_delete_with_triggers(new_stmt)

    # -- INSTEAD OF triggers (on views) --

    def _exec_instead_of_insert(self, stmt: InsertStmt,
                                triggers: List[TriggerDefinition]) -> ResultSet:
        """Execute INSTEAD OF INSERT trigger on a view."""
        # Build the NEW row from the INSERT values
        # For views, column names come from the view definition
        vdef = self.view_catalog.get_view(stmt.table_name)
        if vdef and vdef.columns:
            col_names = vdef.columns
        elif stmt.columns:
            col_names = stmt.columns
        else:
            # Derive from view's SELECT
            col_names = []
            for se in vdef.query.columns:
                if se.alias:
                    col_names.append(se.alias)
                elif isinstance(se.expr, SqlColumnRef):
                    col_names.append(se.expr.column)
                else:
                    col_names.append(f"col_{len(col_names)}")

        count = 0
        for values in stmt.values_list:
            new_row = {}
            for i, col in enumerate(col_names):
                if i < len(values):
                    new_row[col] = self._eval_sql_value(values[i])
                else:
                    new_row[col] = None

            for tdef in triggers:
                if self.trigger_executor.check_when_condition(tdef, None, new_row):
                    try:
                        self.trigger_executor.execute_trigger(
                            tdef, old_row=None, new_row=new_row
                        )
                    except TriggerSignal as e:
                        raise DatabaseError(
                            f"Trigger '{tdef.name}': {e.message}" if e.message
                            else f"Trigger '{tdef.name}' raised SQLSTATE {e.sqlstate}"
                        )
            count += 1

        return ResultSet(columns=[], rows=[], message=f"INSERT {count}",
                       rows_affected=count)

    def _exec_instead_of_update(self, stmt: UpdateStmt,
                                triggers: List[TriggerDefinition]) -> ResultSet:
        """Execute INSTEAD OF UPDATE trigger on a view."""
        # Get current view rows that match WHERE
        vdef = self.view_catalog.get_view(stmt.table_name)
        select = SelectStmt(
            columns=[SelectExpr(expr=SqlStar(), alias=None)],
            from_table=TableRef(table_name=stmt.table_name, alias=None),
            joins=[], where=stmt.where, group_by=[], having=None,
            order_by=None, limit=None, offset=None, distinct=False
        )
        result = self._execute_view_stmt(select)

        count = 0
        for row_vals in result.rows:
            old_row = {}
            for i, col in enumerate(result.columns):
                old_row[col] = row_vals[i]

            new_row = dict(old_row)
            for col, val_expr in stmt.assignments:
                if isinstance(val_expr, SqlLiteral):
                    new_row[col] = val_expr.value
                else:
                    # Evaluate against old row
                    qe_row = Row(old_row)
                    qe_expr = self.compiler._sql_to_qe_expr(val_expr)
                    new_row[col] = eval_expr(qe_expr, qe_row)

            for tdef in triggers:
                if self.trigger_executor.check_when_condition(tdef, old_row, new_row):
                    try:
                        self.trigger_executor.execute_trigger(
                            tdef, old_row=old_row, new_row=new_row
                        )
                    except TriggerSignal as e:
                        raise DatabaseError(
                            f"Trigger '{tdef.name}': {e.message}" if e.message
                            else f"Trigger '{tdef.name}' raised SQLSTATE {e.sqlstate}"
                        )
            count += 1

        return ResultSet(columns=[], rows=[], message=f"UPDATE {count}",
                       rows_affected=count)

    def _exec_instead_of_delete(self, stmt: DeleteStmt,
                                triggers: List[TriggerDefinition]) -> ResultSet:
        """Execute INSTEAD OF DELETE trigger on a view."""
        vdef = self.view_catalog.get_view(stmt.table_name)
        select = SelectStmt(
            columns=[SelectExpr(expr=SqlStar(), alias=None)],
            from_table=TableRef(table_name=stmt.table_name, alias=None),
            joins=[], where=stmt.where, group_by=[], having=None,
            order_by=None, limit=None, offset=None, distinct=False
        )
        result = self._execute_view_stmt(select)

        count = 0
        for row_vals in result.rows:
            old_row = {}
            for i, col in enumerate(result.columns):
                old_row[col] = row_vals[i]

            for tdef in triggers:
                if self.trigger_executor.check_when_condition(tdef, old_row, None):
                    try:
                        self.trigger_executor.execute_trigger(
                            tdef, old_row=old_row, new_row=None
                        )
                    except TriggerSignal as e:
                        raise DatabaseError(
                            f"Trigger '{tdef.name}': {e.message}" if e.message
                            else f"Trigger '{tdef.name}' raised SQLSTATE {e.sqlstate}"
                        )
            count += 1

        return ResultSet(columns=[], rows=[], message=f"DELETE {count}",
                       rows_affected=count)
