# API Setup and Usage

This README provides the steps to set up and run the API locally, as well as interact with the available endpoints.

## Prerequisites

Before starting, ensure you have the following installed:
- Python 3.x
- pip (Python's package installer)
- [Ollama instructions](OLLAMA.md)

## Setup Instructions

### 1. Create and Activate a Virtual Environment

To isolate the project’s dependencies, create a virtual environment. This will ensure that any Python packages required for this project do not interfere with other projects.

#### **Windows:**
```bash
python -m venv env
.\env\Scripts\activate
```
#### **Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### 2. Install Dependencies

Once the environment is set and activated, you must install the dependencies to run the API

```bash
pip install -r requirements.txt
```

### 3. Run the Flask API

Now that the dependencies are installed, you can run the API.
Make sure to navigate to the right folder
```bash
flask run
```