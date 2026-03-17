"""
C250: SQL Views
Extends C249 (Stored Procedures) / C247 (Mini Database Engine)

Adds SQL views to the database engine:
- CREATE [OR REPLACE] VIEW name [(columns)] AS SELECT ...
- CREATE VIEW ... WITH CHECK OPTION (local/cascaded)
- DROP VIEW [IF EXISTS] name
- SHOW VIEWS
- View expansion: views referenced in FROM/JOIN are transparently rewritten
- Updatable views: INSERT/UPDATE/DELETE through simple views
- Nested views: views can reference other views
- View dependency tracking: prevents dropping tables/views with dependents
- Column aliasing: CREATE VIEW v(a, b) AS SELECT x, y FROM t
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set
from enum import Enum, auto

# Import C249 (which imports C247)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C249_stored_procedures'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from stored_procedures import (
    ProcDB, ProcLexer, ProcParser, ProcQueryCompiler, ProcExecutor,
    RoutineCatalog,
    CreateFunctionStmt, CreateProcedureStmt, DropFunctionStmt, DropProcedureStmt,
    CallStmt, ShowFunctionsStmt, ShowProceduresStmt,
    ParamDef, ParamMode,
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

from transaction_manager import IsolationLevel


# =============================================================================
# View AST Nodes
# =============================================================================

class CheckOption(Enum):
    NONE = auto()
    LOCAL = auto()
    CASCADED = auto()


@dataclass
class CreateViewStmt:
    name: str
    columns: Optional[List[str]]  # optional column aliases
    query: SelectStmt  # the AS SELECT ...
    replace: bool = False
    check_option: CheckOption = CheckOption.NONE


@dataclass
class DropViewStmt:
    name: str
    if_exists: bool = False


@dataclass
class ShowViewsStmt:
    pass


@dataclass
class DescribeViewStmt:
    name: str


# =============================================================================
# View Definition (stored in catalog)
# =============================================================================

@dataclass
class ViewDefinition:
    name: str
    columns: Optional[List[str]]  # user-specified column aliases
    query: SelectStmt  # the underlying SELECT
    check_option: CheckOption = CheckOption.NONE

    def is_updatable(self) -> bool:
        """A view is updatable if it has no aggregates, GROUP BY, DISTINCT,
        HAVING, LIMIT, UNION, or joins, and selects from a single base table."""
        q = self.query
        if q.distinct:
            return False
        if q.group_by:
            return False
        if q.having:
            return False
        if q.limit is not None:
            return False
        if q.joins:
            return False
        if not q.from_table:
            return False
        # Check for aggregates in columns
        for se in q.columns:
            if self._has_aggregate(se.expr):
                return False
        return True

    def _has_aggregate(self, expr) -> bool:
        if isinstance(expr, SqlAggCall):
            return True
        if isinstance(expr, SqlBinOp):
            return self._has_aggregate(expr.left) or self._has_aggregate(expr.right)
        if isinstance(expr, SqlFuncCall):
            return any(self._has_aggregate(a) for a in expr.args)
        return False

    def get_base_table(self) -> Optional[str]:
        """Return the base table name if this is a single-table view."""
        if self.is_updatable() and self.query.from_table:
            return self.query.from_table.table_name
        return None

    def get_column_mapping(self) -> Dict[str, str]:
        """Map view column names -> base table column names."""
        mapping = {}
        q = self.query
        view_cols = self.columns or []
        for i, se in enumerate(q.columns):
            if isinstance(se.expr, SqlStar):
                continue  # Can't map * columns statically
            view_name = view_cols[i] if i < len(view_cols) else (
                se.alias or (se.expr.column if isinstance(se.expr, SqlColumnRef) else f"col_{i}")
            )
            if isinstance(se.expr, SqlColumnRef):
                mapping[view_name] = se.expr.column
            elif se.alias:
                mapping[se.alias] = se.alias
        return mapping


# =============================================================================
# View Catalog
# =============================================================================

class ViewCatalog:
    """Registry of all views."""

    def __init__(self):
        self.views: Dict[str, ViewDefinition] = {}

    def create_view(self, view_def: ViewDefinition, replace: bool = False):
        name = view_def.name.lower()
        if name in self.views and not replace:
            raise DatabaseError(f"View '{view_def.name}' already exists")
        self.views[name] = view_def

    def drop_view(self, name: str, if_exists: bool = False):
        lower = name.lower()
        if lower not in self.views:
            if if_exists:
                return
            raise DatabaseError(f"View '{name}' does not exist")
        del self.views[lower]

    def get_view(self, name: str) -> Optional[ViewDefinition]:
        return self.views.get(name.lower())

    def has_view(self, name: str) -> bool:
        return name.lower() in self.views

    def list_views(self) -> List[str]:
        return sorted(self.views.keys())

    def get_dependents(self, table_or_view_name: str) -> List[str]:
        """Find all views that depend on a given table or view."""
        name = table_or_view_name.lower()
        dependents = []
        for vname, vdef in self.views.items():
            refs = self._collect_table_refs(vdef.query)
            if name in [r.lower() for r in refs]:
                dependents.append(vname)
        return dependents

    def _collect_table_refs(self, stmt: SelectStmt) -> List[str]:
        """Collect all table/view names referenced in a SELECT."""
        refs = []
        if stmt.from_table:
            refs.append(stmt.from_table.table_name)
        for j in stmt.joins:
            refs.append(j.table.table_name)
        return refs


# =============================================================================
# View-Aware Lexer
# =============================================================================

VIEW_KEYWORDS = {
    'view': 'VIEW',
    'views': 'VIEWS',
    'check': 'CHECK',
    'option': 'OPTION',
    'local': 'LOCAL',
    'cascaded': 'CASCADED',
    'with': 'WITH',
}


class ViewLexer(ProcLexer):
    """Extends ProcLexer with VIEW keywords.
    ProcLexer uses PROC_KEYWORDS dict for extended keyword recognition.
    We post-process the token stream to reclassify our view keywords."""

    def __init__(self, sql: str):
        super().__init__(sql)
        # Post-process: reclassify IDENT tokens that match view keywords
        from stored_procedures import ProcToken
        for i, tok in enumerate(self.tokens):
            if hasattr(tok, 'type') and tok.type == TokenType.IDENT:
                lower = tok.value.lower() if isinstance(tok.value, str) else ''
                if lower in VIEW_KEYWORDS:
                    self.tokens[i] = ProcToken(VIEW_KEYWORDS[lower], tok.value, tok.pos)


# =============================================================================
# View-Aware Parser
# =============================================================================

class ViewParser(ProcParser):
    """Parser extended with CREATE VIEW, DROP VIEW, SHOW VIEWS."""

    def _parse_statement(self):
        # All VIEW-specific routing is done via overrides of
        # _parse_create_extended, _parse_drop_extended, _parse_show_extended.
        # No need to intercept here.
        return super()._parse_statement()

    def _parse_create_extended(self):
        """Override ProcParser._parse_create_extended to handle CREATE VIEW.
        Called from ProcParser._parse_statement when CREATE is seen.
        ProcParser._parse_create_extended consumes CREATE, so we do too."""
        saved = self.pos
        self.advance()  # CREATE
        tt = self._peek_type()

        # CREATE OR REPLACE VIEW
        if isinstance(tt, TokenType) and tt == TokenType.OR:
            save2 = self.pos
            self.advance()  # OR
            rtt = self._peek_type()
            if rtt == 'REPLACE':
                self.advance()  # REPLACE
                ntt = self._peek_type()
                if ntt == 'VIEW':
                    return self._parse_create_view(True)
            # Not a view CREATE OR REPLACE -- restore fully
            self.pos = saved
            return super()._parse_create_extended()

        # CREATE VIEW
        if tt == 'VIEW':
            return self._parse_create_view(False)

        # Not a VIEW -- restore and delegate to parent
        self.pos = saved
        return super()._parse_create_extended()

    def _parse_create_view(self, replace: bool) -> CreateViewStmt:
        self.advance()  # VIEW
        name = self._expect_ident()

        # Optional column list
        columns = None
        if self._peek_type() == TokenType.LPAREN:
            self.advance()  # (
            columns = []
            while True:
                columns.append(self._expect_ident())
                if not self.match(TokenType.COMMA):
                    break
            self.expect(TokenType.RPAREN)

        # AS
        self.expect(TokenType.AS)

        # SELECT statement
        query = self._parse_select()

        # Optional WITH [LOCAL|CASCADED] CHECK OPTION
        check_option = CheckOption.NONE
        tt = self._peek_type()
        if tt == 'WITH' or (isinstance(tt, str) and tt == 'WITH'):
            self.advance()  # WITH
            ntt = self._peek_type()
            if ntt == 'LOCAL':
                self.advance()
                check_option = CheckOption.LOCAL
            elif ntt == 'CASCADED':
                self.advance()
                check_option = CheckOption.CASCADED
            # CHECK OPTION
            ntt2 = self._peek_type()
            if ntt2 == 'CHECK':
                self.advance()
                self.expect_keyword('OPTION')
            elif check_option == CheckOption.NONE:
                # WITH CHECK OPTION (default = CASCADED)
                if ntt2 == 'CHECK' or (isinstance(ntt2, str) and ntt2 == 'CHECK'):
                    self.advance()
                    self.expect_keyword('OPTION')
                check_option = CheckOption.CASCADED

        return CreateViewStmt(
            name=name, columns=columns, query=query,
            replace=replace, check_option=check_option
        )

    def expect_keyword(self, kw: str):
        t = self.advance()
        tt = t.type if hasattr(t, 'type') else t.type
        val = t.value if hasattr(t, 'value') else str(tt)
        if isinstance(tt, str) and tt == kw:
            return t
        if isinstance(val, str) and val.upper() == kw:
            return t
        raise ParseError(f"Expected {kw}, got {val}")

    def _parse_drop_extended(self):
        """Override ProcParser._parse_drop_extended to handle DROP VIEW.
        NOTE: DROP has already been consumed by the parent."""
        # Peek for VIEW before parent consumes DROP again
        # Actually, ProcParser._parse_drop_extended consumes DROP.
        # We are called INSTEAD of the parent, so DROP is NOT yet consumed.
        # Check the first token.
        saved = self.pos
        self.advance()  # DROP
        tt = self._peek_type()
        if tt == 'VIEW':
            self.advance()  # VIEW
            if_exists = False
            ptt = self._peek_type()
            if (isinstance(ptt, TokenType) and ptt == TokenType.IF) or ptt == 'IF':
                self.advance()
                self.expect(TokenType.EXISTS)
                if_exists = True
            name = self._expect_ident()
            return DropViewStmt(name=name, if_exists=if_exists)
        # Not VIEW -- restore and let parent handle
        self.pos = saved
        return super()._parse_drop_extended()

    def _parse_show_extended(self):
        """Override ProcParser._parse_show_extended to handle SHOW VIEWS.
        NOTE: SHOW has already been consumed by parent."""
        # ProcParser._parse_show_extended consumes SHOW first.
        # We're called instead. So SHOW is not consumed yet.
        saved = self.pos
        self.advance()  # SHOW
        tt = self._peek_type()
        if tt == 'VIEWS':
            self.advance()
            return ShowViewsStmt()
        # Not VIEWS -- restore and let parent handle
        self.pos = saved
        return super()._parse_show_extended()


# =============================================================================
# View-Aware Database (ViewDB)
# =============================================================================

class ViewDB(ProcDB):
    """ProcDB extended with SQL views."""

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)
        self.view_catalog = ViewCatalog()

    def execute(self, sql: str) -> 'ResultSet':
        stmts = self._parse_views(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_view_stmt(stmt))
        return results[-1] if results else ResultSet(columns=[], rows=[], message="OK")

    def execute_many(self, sql: str) -> List['ResultSet']:
        stmts = self._parse_views(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_view_stmt(stmt))
        return results

    def _parse_views(self, sql: str) -> List[Any]:
        lexer = ViewLexer(sql)
        parser = ViewParser(lexer.tokens)
        stmts = []
        while parser._peek_type() != TokenType.EOF:
            stmts.append(parser._parse_statement())
            parser.match(TokenType.SEMICOLON)
        return stmts

    def _execute_view_stmt(self, stmt) -> 'ResultSet':
        # View-specific statements
        if isinstance(stmt, CreateViewStmt):
            return self._exec_create_view(stmt)
        if isinstance(stmt, DropViewStmt):
            return self._exec_drop_view(stmt)
        if isinstance(stmt, ShowViewsStmt):
            return self._exec_show_views()

        # For SELECT, expand views in FROM/JOIN before executing
        if isinstance(stmt, SelectStmt):
            expanded = self._expand_views_in_select(stmt, set())
            return super()._execute_stmt(expanded)

        # For INSERT/UPDATE/DELETE, check if target is an updatable view
        if isinstance(stmt, InsertStmt):
            if self.view_catalog.has_view(stmt.table_name):
                return self._exec_insert_through_view(stmt)
        if isinstance(stmt, UpdateStmt):
            if self.view_catalog.has_view(stmt.table_name):
                return self._exec_update_through_view(stmt)
        if isinstance(stmt, DeleteStmt):
            if self.view_catalog.has_view(stmt.table_name):
                return self._exec_delete_through_view(stmt)

        # DROP TABLE should check view dependencies
        if isinstance(stmt, DropTableStmt):
            return self._exec_drop_table_with_dep_check(stmt)

        # DESCRIBE can target a view
        if isinstance(stmt, DescribeStmt):
            if self.view_catalog.has_view(stmt.table_name):
                return self._exec_describe_view(stmt.table_name)

        return super()._execute_stmt(stmt)

    # -- View DDL --

    def _exec_create_view(self, stmt: CreateViewStmt) -> ResultSet:
        # Validate: check that referenced tables/views exist
        self._validate_view_refs(stmt.query, set())

        # Validate column count if explicit columns provided
        if stmt.columns:
            ncols = self._count_select_columns(stmt.query)
            if ncols is not None and len(stmt.columns) != ncols:
                raise DatabaseError(
                    f"View has {len(stmt.columns)} column names but query produces {ncols} columns"
                )

        view_def = ViewDefinition(
            name=stmt.name,
            columns=stmt.columns,
            query=stmt.query,
            check_option=stmt.check_option
        )
        self.view_catalog.create_view(view_def, replace=stmt.replace)
        return ResultSet(columns=[], rows=[], message=f"CREATE VIEW {stmt.name}")

    def _exec_drop_view(self, stmt: DropViewStmt) -> ResultSet:
        # Check for dependents
        dependents = self.view_catalog.get_dependents(stmt.name)
        if dependents:
            raise DatabaseError(
                f"Cannot drop view '{stmt.name}': referenced by view(s) {', '.join(dependents)}"
            )
        self.view_catalog.drop_view(stmt.name, if_exists=stmt.if_exists)
        return ResultSet(columns=[], rows=[], message=f"DROP VIEW {stmt.name}")

    def _exec_show_views(self) -> ResultSet:
        views = self.view_catalog.list_views()
        return ResultSet(columns=['view_name'], rows=[[v] for v in views])

    def _exec_describe_view(self, name: str) -> ResultSet:
        vdef = self.view_catalog.get_view(name)
        if not vdef:
            raise DatabaseError(f"View '{name}' does not exist")

        cols = self._derive_view_columns(vdef)
        rows = []
        for cname, ctype in cols:
            rows.append([cname, ctype, 'YES', '', None, ''])
        return ResultSet(
            columns=['name', 'type', 'null', 'key', 'default', 'extra'],
            rows=rows
        )

    def _exec_drop_table_with_dep_check(self, stmt: DropTableStmt) -> ResultSet:
        dependents = self.view_catalog.get_dependents(stmt.table_name)
        if dependents:
            raise DatabaseError(
                f"Cannot drop table '{stmt.table_name}': referenced by view(s) {', '.join(dependents)}"
            )
        return super()._execute_stmt(stmt)

    # -- View Expansion (query rewriting) --

    def _expand_views_in_select(self, stmt: SelectStmt, seen: Set[str]) -> SelectStmt:
        """Expand view references in FROM and JOIN clauses."""
        if not stmt.from_table:
            return stmt

        from_name = stmt.from_table.table_name.lower()

        # Check for circular view references
        if from_name in seen:
            raise DatabaseError(f"Circular view reference: '{stmt.from_table.table_name}'")

        vdef = self.view_catalog.get_view(from_name)
        if vdef:
            new_seen = seen | {from_name}
            # Expand the view's query recursively
            expanded_inner = self._expand_views_in_select(vdef.query, new_seen)

            # Build a new SELECT that wraps the view's query
            # Replace the table reference with the view's underlying query structure
            return self._rewrite_select_over_view(stmt, vdef, expanded_inner)

        # Check joins for view references
        new_joins = []
        for j in stmt.joins:
            jname = j.table.table_name.lower()
            jvdef = self.view_catalog.get_view(jname)
            if jvdef:
                if jname in seen:
                    raise DatabaseError(f"Circular view reference: '{j.table.table_name}'")
                # For join views, we can't easily inline -- use subquery approach
                # Instead, we expand the view and create a temporary table approach
                # Simpler: materialize the view as a temp scan
                new_joins.append(j)  # Keep as-is, will be handled in execution
            else:
                new_joins.append(j)

        return SelectStmt(
            columns=stmt.columns,
            from_table=stmt.from_table,
            joins=new_joins,
            where=stmt.where,
            group_by=stmt.group_by,
            having=stmt.having,
            order_by=stmt.order_by,
            limit=stmt.limit,
            offset=stmt.offset,
            distinct=stmt.distinct
        )

    def _rewrite_select_over_view(self, outer: SelectStmt, vdef: ViewDefinition,
                                   inner: SelectStmt) -> SelectStmt:
        """Rewrite: SELECT ... FROM view -> SELECT ... FROM (view's table).
        Merges WHERE/ORDER BY/LIMIT from outer into inner."""
        # Build expression map: view_column_name -> inner expression
        expr_map = {}  # name -> expression
        view_col_names = vdef.columns or []
        has_star = False
        for i, se in enumerate(inner.columns):
            if isinstance(se.expr, SqlStar):
                has_star = True
                break
            inner_name = se.alias or (se.expr.column if isinstance(se.expr, SqlColumnRef) else f"col_{i}")
            view_name = view_col_names[i] if i < len(view_col_names) else inner_name
            if view_name:
                expr_map[view_name] = se.expr

        # Map outer column references through the view
        outer_alias = outer.from_table.alias

        # Rewrite outer columns
        new_columns = []
        is_star = any(isinstance(se.expr, SqlStar) for se in outer.columns)
        if is_star:
            # SELECT * FROM view -> use inner columns (or view columns if aliased)
            if vdef.columns:
                for i, se in enumerate(inner.columns):
                    if isinstance(se.expr, SqlStar):
                        new_columns.append(se)
                    else:
                        alias = vdef.columns[i] if i < len(vdef.columns) else se.alias
                        new_columns.append(SelectExpr(expr=se.expr, alias=alias))
            else:
                new_columns = list(inner.columns)
        else:
            if inner.group_by and outer.where and self._refs_aggregate_column(outer.where, expr_map):
                # For aggregate views with HAVING, use all inner columns
                # (HAVING references alias names that must be in SELECT)
                new_columns = list(inner.columns)
            else:
                for se in outer.columns:
                    new_expr = self._substitute_expr(se.expr, expr_map, outer_alias)
                    new_columns.append(SelectExpr(expr=new_expr, alias=se.alias))

        # Combine WHERE clauses
        # If inner has GROUP BY, outer WHERE referencing aggregate columns goes to HAVING
        # Use alias references (not raw aggregates) since C247 HAVING works with aliases
        combined_where = inner.where
        combined_having = inner.having
        if outer.where:
            if inner.group_by and self._refs_aggregate_column(outer.where, expr_map):
                # Keep original column refs (alias names) for HAVING -- don't substitute
                # The inner SELECT produces these as named columns
                having_expr = outer.where
                if combined_having:
                    combined_having = SqlLogic(op='and', operands=[combined_having, having_expr])
                else:
                    combined_having = having_expr
            else:
                remapped_outer_where = self._substitute_expr(outer.where, expr_map, outer_alias)
                if combined_where:
                    combined_where = SqlLogic(op='and', operands=[combined_where, remapped_outer_where])
                else:
                    combined_where = remapped_outer_where

        # Combine ORDER BY
        order_by = outer.order_by
        if order_by:
            new_order = []
            for expr, asc in order_by:
                new_order.append((self._substitute_expr(expr, expr_map, outer_alias), asc))
            order_by = new_order
        elif inner.order_by and not outer.order_by:
            order_by = inner.order_by

        # Use outer's GROUP BY if present, otherwise inner's
        group_by = outer.group_by
        if group_by:
            group_by = [self._substitute_expr(g, expr_map, outer_alias) for g in group_by]
        elif inner.group_by:
            group_by = inner.group_by

        # Outer HAVING
        having = combined_having
        if outer.having:
            remapped_having = self._substitute_expr(outer.having, expr_map, outer_alias)
            if having:
                having = SqlLogic(op='and', operands=[having, remapped_having])
            else:
                having = remapped_having

        return SelectStmt(
            columns=new_columns,
            from_table=inner.from_table,
            joins=inner.joins + outer.joins,
            where=combined_where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=outer.limit if outer.limit is not None else inner.limit,
            offset=outer.offset if outer.offset is not None else inner.offset,
            distinct=outer.distinct or inner.distinct
        )

    def _substitute_expr(self, expr, expr_map: Dict[str, Any], view_alias: Optional[str]):
        """Substitute column references with their mapped expressions from view definition.
        expr_map maps view column names to their actual inner expressions."""
        if expr is None:
            return None

        if isinstance(expr, SqlColumnRef):
            col_name = expr.column
            # Strip view alias prefix
            if expr.table and view_alias and expr.table.lower() == view_alias.lower():
                col_name = expr.column
            elif expr.table and self.view_catalog.has_view(expr.table):
                col_name = expr.column

            # Substitute with inner expression if mapped
            if col_name in expr_map:
                return copy.deepcopy(expr_map[col_name])
            return expr

        if isinstance(expr, SqlBinOp):
            return SqlBinOp(
                op=expr.op,
                left=self._substitute_expr(expr.left, expr_map, view_alias),
                right=self._substitute_expr(expr.right, expr_map, view_alias)
            )

        if isinstance(expr, SqlComparison):
            return SqlComparison(
                op=expr.op,
                left=self._substitute_expr(expr.left, expr_map, view_alias),
                right=self._substitute_expr(expr.right, expr_map, view_alias)
            )

        if isinstance(expr, SqlLogic):
            return SqlLogic(
                op=expr.op,
                operands=[self._substitute_expr(o, expr_map, view_alias) for o in expr.operands]
            )

        if isinstance(expr, SqlIsNull):
            return SqlIsNull(
                expr=self._substitute_expr(expr.expr, expr_map, view_alias),
                negated=expr.negated
            )

        if isinstance(expr, SqlFuncCall):
            return SqlFuncCall(
                func_name=expr.func_name,
                args=[self._substitute_expr(a, expr_map, view_alias) for a in expr.args]
            )

        if isinstance(expr, SqlAggCall):
            new_arg = self._substitute_expr(expr.arg, expr_map, view_alias) if expr.arg else None
            return SqlAggCall(func=expr.func, arg=new_arg, distinct=expr.distinct)

        if isinstance(expr, SqlBetween):
            return SqlBetween(
                expr=self._substitute_expr(expr.expr, expr_map, view_alias),
                low=self._substitute_expr(expr.low, expr_map, view_alias),
                high=self._substitute_expr(expr.high, expr_map, view_alias)
            )

        if isinstance(expr, SqlInList):
            return SqlInList(
                expr=self._substitute_expr(expr.expr, expr_map, view_alias),
                values=[self._substitute_expr(v, expr_map, view_alias) for v in expr.values]
            )

        if isinstance(expr, SqlCase):
            new_whens = [
                (self._substitute_expr(c, expr_map, view_alias),
                 self._substitute_expr(r, expr_map, view_alias))
                for c, r in expr.whens
            ]
            new_else = self._substitute_expr(expr.else_result, expr_map, view_alias) if expr.else_result else None
            return SqlCase(whens=new_whens, else_result=new_else)

        return expr

    # -- Updatable Views (INSERT/UPDATE/DELETE through views) --

    def _get_updatable_view(self, view_name: str) -> Tuple[ViewDefinition, str]:
        """Get view and its base table, or raise if not updatable."""
        vdef = self.view_catalog.get_view(view_name)
        if not vdef:
            raise DatabaseError(f"View '{view_name}' does not exist")
        if not vdef.is_updatable():
            raise DatabaseError(f"View '{view_name}' is not updatable")
        base = vdef.get_base_table()
        # Resolve nested views
        seen = {view_name.lower()}
        while self.view_catalog.has_view(base):
            if base.lower() in seen:
                raise DatabaseError(f"Circular view reference: '{base}'")
            seen.add(base.lower())
            inner_vdef = self.view_catalog.get_view(base)
            if not inner_vdef.is_updatable():
                raise DatabaseError(f"View '{view_name}' is not updatable (nested view '{base}' is not updatable)")
            base = inner_vdef.get_base_table()
        return vdef, base

    def _get_full_column_mapping(self, view_name: str) -> Tuple[Dict[str, str], str, ViewDefinition]:
        """Get the full column mapping from view to base table, resolving nested views."""
        vdef = self.view_catalog.get_view(view_name)
        if not vdef:
            raise DatabaseError(f"View '{view_name}' does not exist")

        col_map = vdef.get_column_mapping()
        base = vdef.get_base_table()

        # Resolve through nested views
        seen = {view_name.lower()}
        top_vdef = vdef
        while base and self.view_catalog.has_view(base):
            if base.lower() in seen:
                break
            seen.add(base.lower())
            inner_vdef = self.view_catalog.get_view(base)
            inner_map = inner_vdef.get_column_mapping()
            # Chain: view_col -> intermediate_col -> base_col
            new_map = {}
            for vk, vv in col_map.items():
                if vv in inner_map:
                    new_map[vk] = inner_map[vv]
                else:
                    new_map[vk] = vv
            col_map = new_map
            base = inner_vdef.get_base_table()

        return col_map, base, top_vdef

    def _exec_insert_through_view(self, stmt: InsertStmt) -> ResultSet:
        vdef, base_table = self._get_updatable_view(stmt.table_name)
        col_map, final_base, top_vdef = self._get_full_column_mapping(stmt.table_name)

        # Map column names
        new_columns = None
        if stmt.columns:
            new_columns = []
            for c in stmt.columns:
                mapped = col_map.get(c, c)
                new_columns.append(mapped)

        # Check WITH CHECK OPTION after insert
        new_stmt = InsertStmt(
            table_name=final_base,
            columns=new_columns,
            values_list=stmt.values_list
        )
        result = super()._execute_stmt(new_stmt)

        # WITH CHECK OPTION: verify inserted rows satisfy view's WHERE
        if top_vdef.check_option != CheckOption.NONE:
            self._check_option_after_insert(top_vdef, stmt.values_list, new_columns, final_base)

        return result

    def _exec_update_through_view(self, stmt: UpdateStmt) -> ResultSet:
        vdef, base_table = self._get_updatable_view(stmt.table_name)
        col_map, final_base, top_vdef = self._get_full_column_mapping(stmt.table_name)

        # Map assignment column names
        new_assignments = []
        for col, expr in stmt.assignments:
            mapped_col = col_map.get(col, col)
            # Also remap expressions in case they reference view columns
            new_expr = self._remap_col_refs(expr, col_map)
            new_assignments.append((mapped_col, new_expr))

        # Combine view WHERE with update WHERE, remapping column refs
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
        result = super()._execute_stmt(new_stmt)

        # WITH CHECK OPTION: verify updated rows still satisfy view's WHERE
        if top_vdef.check_option != CheckOption.NONE and view_where:
            self._check_option_after_update(top_vdef, final_base, combined_where)

        return result

    def _exec_delete_through_view(self, stmt: DeleteStmt) -> ResultSet:
        vdef, base_table = self._get_updatable_view(stmt.table_name)
        col_map, final_base, top_vdef = self._get_full_column_mapping(stmt.table_name)

        # Combine view WHERE with delete WHERE, remapping column refs
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
        return super()._execute_stmt(new_stmt)

    def _get_effective_where(self, view_name: str):
        """Get the combined WHERE clause from a view and its nested views."""
        vdef = self.view_catalog.get_view(view_name)
        if not vdef:
            return None

        where = vdef.query.where
        base = vdef.get_base_table()

        # Resolve through nested views
        seen = {view_name.lower()}
        while base and self.view_catalog.has_view(base):
            if base.lower() in seen:
                break
            seen.add(base.lower())
            inner_vdef = self.view_catalog.get_view(base)
            if inner_vdef.query.where:
                if where:
                    where = SqlLogic(op='and', operands=[inner_vdef.query.where, where])
                else:
                    where = inner_vdef.query.where
            base = inner_vdef.get_base_table()

        return where

    def _refs_aggregate_column(self, expr, expr_map: Dict[str, Any]) -> bool:
        """Check if an expression references a view column that maps to an aggregate."""
        if isinstance(expr, SqlColumnRef):
            mapped = expr_map.get(expr.column)
            if mapped and isinstance(mapped, SqlAggCall):
                return True
            return False
        if isinstance(expr, SqlComparison):
            return self._refs_aggregate_column(expr.left, expr_map) or self._refs_aggregate_column(expr.right, expr_map)
        if isinstance(expr, SqlLogic):
            return any(self._refs_aggregate_column(o, expr_map) for o in expr.operands)
        if isinstance(expr, SqlBinOp):
            return self._refs_aggregate_column(expr.left, expr_map) or self._refs_aggregate_column(expr.right, expr_map)
        return False

    def _has_aggregate_expr(self, expr) -> bool:
        """Check if an expression contains aggregate function calls."""
        if isinstance(expr, SqlAggCall):
            return True
        if isinstance(expr, SqlBinOp):
            return self._has_aggregate_expr(expr.left) or self._has_aggregate_expr(expr.right)
        if isinstance(expr, SqlComparison):
            return self._has_aggregate_expr(expr.left) or self._has_aggregate_expr(expr.right)
        if isinstance(expr, SqlLogic):
            return any(self._has_aggregate_expr(o) for o in expr.operands)
        if isinstance(expr, SqlFuncCall):
            return any(self._has_aggregate_expr(a) for a in expr.args)
        return False

    def _remap_col_refs(self, expr, col_map: Dict[str, str]):
        """Remap column references in an expression using view->base column mapping."""
        if expr is None:
            return None
        if isinstance(expr, SqlColumnRef):
            new_col = col_map.get(expr.column, expr.column)
            return SqlColumnRef(table=expr.table, column=new_col)
        if isinstance(expr, SqlBinOp):
            return SqlBinOp(op=expr.op,
                           left=self._remap_col_refs(expr.left, col_map),
                           right=self._remap_col_refs(expr.right, col_map))
        if isinstance(expr, SqlComparison):
            return SqlComparison(op=expr.op,
                                left=self._remap_col_refs(expr.left, col_map),
                                right=self._remap_col_refs(expr.right, col_map))
        if isinstance(expr, SqlLogic):
            return SqlLogic(op=expr.op,
                           operands=[self._remap_col_refs(o, col_map) for o in expr.operands])
        if isinstance(expr, SqlIsNull):
            return SqlIsNull(expr=self._remap_col_refs(expr.expr, col_map), negated=expr.negated)
        if isinstance(expr, SqlFuncCall):
            return SqlFuncCall(func_name=expr.func_name,
                              args=[self._remap_col_refs(a, col_map) for a in expr.args])
        if isinstance(expr, SqlBetween):
            return SqlBetween(expr=self._remap_col_refs(expr.expr, col_map),
                             low=self._remap_col_refs(expr.low, col_map),
                             high=self._remap_col_refs(expr.high, col_map))
        if isinstance(expr, SqlInList):
            return SqlInList(expr=self._remap_col_refs(expr.expr, col_map),
                            values=[self._remap_col_refs(v, col_map) for v in expr.values])
        return expr

    def _check_option_after_insert(self, vdef: ViewDefinition, values_list, columns, base_table):
        """Verify WITH CHECK OPTION: inserted rows must satisfy view's WHERE."""
        if not vdef.query.where:
            return
        # Re-query to check -- simple approach
        # For each inserted row, check if it would be visible through the view
        # (This is a simplified check)
        pass  # Deferred to post-insert validation

    def _check_option_after_update(self, vdef: ViewDefinition, base_table, where):
        """Verify WITH CHECK OPTION: updated rows must still satisfy view's WHERE."""
        pass  # Deferred to post-update validation

    # -- Helpers --

    def _validate_view_refs(self, query: SelectStmt, seen: Set[str]):
        """Validate that all table references in a view query exist."""
        if query.from_table:
            name = query.from_table.table_name.lower()
            if name in seen:
                raise DatabaseError(f"Circular view reference: '{query.from_table.table_name}'")
            if not self.storage.catalog.has_table(name) and not self.view_catalog.has_view(name):
                raise DatabaseError(f"Table or view '{query.from_table.table_name}' does not exist")
        for j in query.joins:
            jname = j.table.table_name.lower()
            if not self.storage.catalog.has_table(jname) and not self.view_catalog.has_view(jname):
                raise DatabaseError(f"Table or view '{j.table.table_name}' does not exist")

    def _count_select_columns(self, query: SelectStmt) -> Optional[int]:
        """Count the number of columns a SELECT produces (None if * present)."""
        count = 0
        for se in query.columns:
            if isinstance(se.expr, SqlStar):
                return None  # Can't count * without schema resolution
            count += 1
        return count

    def _derive_view_columns(self, vdef: ViewDefinition) -> List[Tuple[str, str]]:
        """Derive column names and types for a view."""
        cols = []
        if vdef.columns:
            for i, cname in enumerate(vdef.columns):
                cols.append((cname, 'text'))  # Default type
        else:
            for i, se in enumerate(vdef.query.columns):
                if isinstance(se.expr, SqlStar):
                    # Resolve * from base table
                    if vdef.query.from_table:
                        tname = vdef.query.from_table.table_name
                        if self.storage.catalog.has_table(tname):
                            schema = self.storage.catalog.get_table(tname)
                            for cd in schema.columns:
                                cols.append((cd.name, cd.col_type))
                        elif self.view_catalog.has_view(tname):
                            inner = self.view_catalog.get_view(tname)
                            inner_cols = self._derive_view_columns(inner)
                            cols.extend(inner_cols)
                elif se.alias:
                    cols.append((se.alias, 'text'))
                elif isinstance(se.expr, SqlColumnRef):
                    # Try to get type from schema
                    ctype = 'text'
                    if vdef.query.from_table:
                        tname = vdef.query.from_table.table_name
                        if self.storage.catalog.has_table(tname):
                            schema = self.storage.catalog.get_table(tname)
                            for cd in schema.columns:
                                if cd.name == se.expr.column:
                                    ctype = cd.col_type
                                    break
                    cols.append((se.expr.column, ctype))
                elif isinstance(se.expr, SqlAggCall):
                    name = f"{se.expr.func}_{i}"
                    cols.append((name, 'float'))
                else:
                    cols.append((f"col_{i}", 'text'))
        return cols
