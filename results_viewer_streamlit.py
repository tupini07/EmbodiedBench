import streamlit as st
import subprocess
from pathlib import Path

st.set_page_config(page_title="Results Viewer", layout="wide")

st.title("EmbodiedBench Results Viewer")

# Add a button to recompute results
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Recompute Results", type="primary"):
        with st.spinner("Running print_results_table.py..."):
            try:
                result = subprocess.run(
                    ["python", "scripts/print_results_table.py"],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent
                )
                if result.returncode == 0:
                    st.success("✅ Results recomputed successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Error running script:\n{result.stderr}")
            except Exception as e:
                st.error(f"❌ Exception occurred: {str(e)}")

with col2:
    st.info("Click the button to regenerate the results file")

# Display the results file
results_file = Path(__file__).parent / "results_pp.txt"

if results_file.exists():
    st.markdown("---")
    st.subheader("📊 Current Results")
    
    # Read and display the file content
    with open(results_file, "r") as f:
        content = f.read()
    
    # Display in a code block for better formatting
    st.markdown(content)
else:
    st.warning("⚠️ results_pp.txt file not found. Click 'Recompute Results' to generate it.")
