install_dependencies:
	pip3.11 install --no-cache-dir -r ./requirements.txt

update_YOLOv8_default_models:
	python3.11 ./Update_YOLOv8_Default_Models.py

run:
	python3.11 -m streamlit run ./main.py

clean:
	pip3.11 cache purge && rm -rf ./__pycache__