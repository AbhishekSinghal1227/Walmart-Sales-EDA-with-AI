def general_question_tool(user_input: str) -> str:
    """
    Step 8: Execute AI-generated Python code and return actual output
    """
    columns = list(df.columns)
    llm_response = llm.invoke(f"""
    You are a data analyst for the Walmart dataset with columns: {columns}

    User Question: "{user_input}"

    Provide:
    1. Python (pandas) code to answer the question for df where df is Walmart.csv
       ⚠️ Important: Include calculation of Sales if not present and assign the final result to _result
    2. SQL query (assume table name = walmart)
    3. Brief explanation in human language

    Return in clear sections:
    Python:
    <code>

    SQL:
    <code>

    Explanation:
    <text>
    """)

    response_text = llm_response.content

    # Extract Python code safely
    match = re.search(r"```python(.*?)```", response_text, re.S)
    if match:
        python_code = match.group(1).strip()
    else:
        try:
            python_code = response_text.split("Python:")[1].split("SQL:")[0].strip(" \n```")
        except:
            python_code = ""

    # Prepend Sales calculation if needed
    sales_calc = """
if 'Sales' not in df.columns:
    df['unit_price'] = df['unit_price'].str.replace('$', '').astype(float)
    df['Sales'] = df['unit_price'] * df['quantity']
"""

    full_code = sales_calc + "\n" + python_code

    # Execute Python code safely
    exec_globals = {'df': df.copy(), '_result': None, 'pd': pd}
    try:
        exec(full_code, exec_globals)
        result = exec_globals.get('_result', None)
        result_text = str(result)
    except Exception as e:
        result_text = f"⚠️ Error executing code: {e}"

    return f"Python:\n{full_code}\n\nSQL + Explanation:\n{response_text.split('SQL:')[1].strip() if 'SQL:' in response_text else ''}\n\nExecution Result:\n{result_text}"
