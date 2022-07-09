# sql-injection-prediction

This python module can be used to differentiate between normal usernames or passwords and SQL-Injection attempts.

## Setup

1. Download this module from this github-repository (git install is not available yet).
2. Add the module to the folder in which your other modules/libraries are kept (numpy, pandas etc.)
3. You should be good to go. Take a look at the usage example to find out how to use this module.

Important note: TensorFlow has to be installed for this module to work.

## Usage example

Import the InjectionPredictor-class (and create the IP-object):
```
from injection_classifier.classifiers import InjectionPredictor
IP = InjectionPredictor()
```

Specify the text which has to be classified
```
text = "' or 1=1 '"
```

Use the `predict`-function to predict whether the text is a SQL-Injection or a normal username/password:
```
result = IP.predict(' or 1=1 ')
print(result)
```

Output:
```
{'prediction': 'SQL Injection', 'confidence': 91.18}
```

You can specify the output by using `print(result['prediction'])` or `print(result['confidence'])`.
