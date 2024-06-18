from sql_metadata import Parser
import re


def remove_table_alias(s):
    tables_aliases = Parser(s).tables_aliases
    new_tables_aliases = {}
    for i in range(1, 11):
        if "t{}".format(i) in tables_aliases.keys():
            new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]

    tables_aliases = new_tables_aliases
    for k, v in tables_aliases.items():
        s = s.replace("as " + k + " ", "")
        s = s.replace(k, v)

    return s

def white_space_fix(s):
    parsed_s = Parser(s)
    s = " ".join([token.value for token in parsed_s.tokens])

    return s


def lower(s):
    in_quotation = False
    out_s = ""
    for char in s:
        if in_quotation:
            out_s += char
        else:
            out_s += char.lower()

        if char == "'":
            if in_quotation:
                in_quotation = False
            else:
                in_quotation = True

    return out_s

# remove ";"
def add_semicolon(s):
    if not s.endswith(";"):
        s = s + ';'
    return s

# double quotation -> single quotation
def double2single(s):
    return s.replace("\"", "'")

def add_asc(s):
    pattern = re.compile(
        r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
    if "order by" in s and "asc" not in s and "desc" not in s:
        for p_str in pattern.findall(s):
            s = s.replace(p_str, p_str + " asc")

    return s

def normalization(sql):
    """
    adapted from https://github.com/RUCKBReasoning/RESDSQL
    :param sql:
    :return:
    """
    formatted_sql = sql.strip()
    formatted_sql = add_semicolon(formatted_sql)
    formatted_sql = double2single(formatted_sql)
    formatted_sql = white_space_fix(formatted_sql)
    formatted_sql = lower(formatted_sql)
    formatted_sql = add_asc(formatted_sql)
    formatted_sql = remove_table_alias(formatted_sql)
    formatted_sql = formatted_sql.replace('\t', "")

    return formatted_sql


def normalize_sql_query(query_str):
    query_str = query_str.strip()
    norm_sql = normalization(query_str)
    return norm_sql


def process_input_question(question_str):
    question_str = question_str.replace("\u2018", "'").replace("\u2019", "'") \
        .replace("\u201c", "'").replace("\u201d", "'").replace('\t', "").strip().replace('\n', '')
    return question_str