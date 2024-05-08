# **Income Prediction API**

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

> [!IMPORTANT]
> This is a project for academic purposes.

This application serves a Machine Learning model to predict wether a person has yearly earnings above 50K USD or not. Trained with the [public Census Income dataset from UCI ML repo](https://archive.ics.uci.edu/dataset/20/census+income).

**Model accuracy is around 82.7%.**

For more information about data exploration, preprocessing, training iterations and model accuracy, see the [Jupyter Notebooks repository](https://github.com/daniel-lujan/ModelosII).

## **Quickstart**

### **Creating virtual environment**

> [!NOTE]
> This is not a required step, although it is highly recommended to use a virtual environment to avoid incompatibility issues with required packages.

```bash
# or pip3
pip install virtualenv
# or python3
python -m venv venv
# or source /venv/bin/activate
./venv/Scripts/activate
```

### **Installing dependencies**

```bash
# or pip3
pip install -r requirements.txt
```

### Run the server

```bash
# or pip3
fastapi run
```

You can then see the available endpoint documentation at [**`http://localhost:8000/docs`**](http://localhost:8000/docs) and try requests from there.
