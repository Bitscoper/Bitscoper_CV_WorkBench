install_dependencies:
	pip3.11 install --no-cache-dir -r requirements.txt

run:
	python3.11 -m streamlit run main.py

clean:
	pip3.11 cache purge && rm -rf __pycache__