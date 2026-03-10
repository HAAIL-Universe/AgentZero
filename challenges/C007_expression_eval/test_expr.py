"""Tests for expression evaluator."""

import unittest
from expr_eval import Evaluator, EvalError, calc, tokenize


class TestArithmetic(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator()

    def test_integer(self):
        assert self.ev.evaluate("42") == 42.0

    def test_float(self):
        assert self.ev.evaluate("3.14") == 3.14

    def test_addition(self):
        assert self.ev.evaluate("2 + 3") == 5.0

    def test_subtraction(self):
        assert self.ev.evaluate("10 - 4") == 6.0

    def test_multiplication(self):
        assert self.ev.evaluate("3 * 7") == 21.0

    def test_division(self):
        assert self.ev.evaluate("15 / 3") == 5.0

    def test_float_division(self):
        assert abs(self.ev.evaluate("7 / 2") - 3.5) < 1e-10

    def test_division_by_zero(self):
        with self.assertRaises(EvalError):
            self.ev.evaluate("1 / 0")


class TestPrecedence(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator()

    def test_mul_before_add(self):
        assert self.ev.evaluate("2 + 3 * 4") == 14.0

    def test_div_before_sub(self):
        assert self.ev.evaluate("10 - 6 / 3") == 8.0

    def test_left_to_right_add_sub(self):
        assert self.ev.evaluate("10 - 3 - 2") == 5.0

    def test_left_to_right_mul_div(self):
        assert self.ev.evaluate("12 / 3 * 2") == 8.0

    def test_complex_precedence(self):
        assert self.ev.evaluate("2 + 3 * 4 - 6 / 2") == 11.0


class TestParentheses(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator()

    def test_simple_parens(self):
        assert self.ev.evaluate("(2 + 3) * 4") == 20.0

    def test_nested_parens(self):
        assert self.ev.evaluate("((2 + 3) * (4 - 1))") == 15.0

    def test_deeply_nested(self):
        assert self.ev.evaluate("(((1 + 2)))") == 3.0

    def test_unmatched_open(self):
        with self.assertRaises(EvalError):
            self.ev.evaluate("(2 + 3")

    def test_unmatched_close(self):
        with self.assertRaises(EvalError):
            self.ev.evaluate("2 + 3)")


class TestUnaryMinus(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator()

    def test_negate_number(self):
        assert self.ev.evaluate("-5") == -5.0

    def test_double_negate(self):
        assert self.ev.evaluate("--5") == 5.0

    def test_negate_in_expression(self):
        assert self.ev.evaluate("3 + -2") == 1.0

    def test_negate_parens(self):
        assert self.ev.evaluate("-(3 + 2)") == -5.0


class TestVariables(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator()

    def test_assign_and_use(self):
        self.ev.evaluate("x = 10")
        assert self.ev.evaluate("x + 5") == 15.0

    def test_assign_returns_value(self):
        assert self.ev.evaluate("x = 42") == 42.0

    def test_multiple_variables(self):
        self.ev.evaluate("a = 3")
        self.ev.evaluate("b = 4")
        assert self.ev.evaluate("a * b") == 12.0

    def test_reassignment(self):
        self.ev.evaluate("x = 1")
        self.ev.evaluate("x = 2")
        assert self.ev.evaluate("x") == 2.0

    def test_variable_in_assignment(self):
        self.ev.evaluate("x = 10")
        self.ev.evaluate("y = x * 2")
        assert self.ev.evaluate("y") == 20.0

    def test_undefined_variable(self):
        with self.assertRaises(EvalError):
            self.ev.evaluate("unknown + 1")

    def test_predefined_variables(self):
        ev = Evaluator({"pi": 3.14159})
        assert abs(ev.evaluate("pi * 2") - 6.28318) < 1e-4

    def test_underscore_variable(self):
        self.ev.evaluate("my_var = 7")
        assert self.ev.evaluate("my_var") == 7.0


class TestSyntaxErrors(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator()

    def test_empty_expression(self):
        with self.assertRaises(EvalError):
            self.ev.evaluate("")

    def test_trailing_operator(self):
        with self.assertRaises(EvalError):
            self.ev.evaluate("5 +")

    def test_double_operator(self):
        with self.assertRaises(EvalError):
            self.ev.evaluate("5 * + 3")

    def test_invalid_character(self):
        with self.assertRaises(EvalError):
            self.ev.evaluate("5 @ 3")

    def test_error_has_position(self):
        try:
            self.ev.evaluate("5 @ 3")
        except EvalError as e:
            assert e.pos == 2


class TestConvenienceFunction(unittest.TestCase):
    def test_calc_basic(self):
        assert calc("2 + 2") == 4.0

    def test_calc_with_vars(self):
        assert calc("x + 1", {"x": 9}) == 10.0


class TestTokenizer(unittest.TestCase):
    def test_simple_tokens(self):
        tokens = tokenize("2 + 3")
        types = [t.type.name for t in tokens]
        assert types == ["NUMBER", "PLUS", "NUMBER", "EOF"]

    def test_decimal_leading_dot(self):
        tokens = tokenize(".5")
        assert tokens[0].value == ".5"


if __name__ == "__main__":
    unittest.main()
