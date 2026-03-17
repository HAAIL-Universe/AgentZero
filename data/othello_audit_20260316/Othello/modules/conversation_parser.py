# modules/conversation_parser.py
"""
Thin adapter so existing code that imports from modules.conversation_parser
can use the real ConversationParser implementation in core.conversation_parser.
"""

from core.conversation_parser import ConversationParser

__all__ = ["ConversationParser"]
