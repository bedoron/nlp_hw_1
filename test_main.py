import unittest
from main import tokenize_word, tokenize_sentence, transform_paragraphs


class TestTokenization(unittest.TestCase):

    def test_enumeration(self):
        text_with_enumeration = "1. שמוליק בוליק בום בום"

        result = tokenize_sentence(text_with_enumeration)
        self.assertEqual(text_with_enumeration, result)

    def test_initials(self):
        initials_in_text = "א.ב.ג."

        result = tokenize_word(initials_in_text, 'middle')
        self.assertEqual([initials_in_text], result)

    def test_initials_next_to_token(self):
        initials_with_token = "א.ב.ג.,"

        result = tokenize_word(initials_with_token, 'middle')

        self.assertEqual(["א.ב.ג.", ","], result)

    def test_initials_next_to_token_with_word_boundary(self):
        initials_with_token = "א.ב.ג., באעסה"

        result = tokenize_sentence(initials_with_token)
        self.assertEqual("א.ב.ג. , באעסה", result)

    def test_initials_next_to_token_with_word_boundary_and_ending(self):
        initials_with_token = "א.ב.ג., באעסה?"

        result = tokenize_sentence(initials_with_token)
        self.assertEqual("א.ב.ג. , באעסה ?", result)

    def test_enumeration_line(self):
        text = "1. אנומרציה בלה, בלה בלה איזה כיף!"
        expected = "1. אנומרציה בלה , בלה בלה איזה כיף !"

        result = tokenize_sentence(text)
        self.assertEqual(expected, result)

    def test_numbering_with_colon(self):
        text = "הפיראט ה - 300: אין לי כח!"
        expected = "הפיראט ה - 300 : אין לי כח !"

        result = tokenize_sentence(text)
        self.assertEqual(expected, result)

    def test_last_sentence_tokenization_issue(self):
        paragraph_text = """כבוד יושב-ראש הכנסת, אדוני שר החינוך, במרכז מדע ודעת למחוננים בשלומי לומדים 351 תלמידים יהודים ודרוזים. במשרד החינוך התקבלה החלטה להעביר את המרכז לכרמיאל, במקום לחזק את שלומי כעיר פיתוח."""
        result = transform_paragraphs("Monkey", [paragraph_text])
        self.assertEqual(3, len(result.split('.'))) # Its stupid :(

    def test_numeral_percent(self):
        text = "הפיתוח, 75% מההערכה"
        expected = "הפיתוח , 75% מההערכה"

        result = transform_paragraphs("Monkey", [text])
        self.assertEqual(expected, result)

    def test_numberz(self):
        text = "1. יש לי 1,000 שטויות בז.ב.ל.."
        expected = "1. יש לי 1,000 שטויות בז.ב.ל. ."

        result = transform_paragraphs("Monkey", [text])
        self.assertEqual(expected, result)

    def test_acronym_with_comma(self):
        text = "איזה ז.ב.מ., מלך!"
        expected = "איזה ז.ב.מ. , מלך !"

        result = transform_paragraphs("Monkey", [text])
        self.assertEqual(expected, result)

    def test_quoted_text_tokenization(self):
        text = 'לבנות את ה"פתח-לנד" ולפעול'
        expected = 'לבנות את ה " פתח-לנד " ולפעול'
        result = transform_paragraphs("Monkey", [text])
        self.assertEqual(expected, result)

    def test_paranthesis(self):
        text = 'איזה קטע (יש פה "סוגריים") עובד.'
        expected = 'איזה קטע ( יש פה " סוגריים " ) עובד .'
        result = transform_paragraphs("Monkey", [text])
        self.assertEqual(expected, result)

    def test_paranthesis_with_numbers_and_question(self):
        text = "טובין (תיקון מס' 27) מי נגד"
        expected = "טובין ( תיקון מס' 27 ) מי נגד"
        result = transform_paragraphs("Monkey", [text])
        self.assertEqual(expected, result)


