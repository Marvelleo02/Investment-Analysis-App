
import streamlit as st
from functools import wraps

def error_boundary(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"🚨 Something went wrong: {str(e)}")
            st.button("🔄 Retry", on_click=st.experimental_rerun)
            return None
    return wrapper
