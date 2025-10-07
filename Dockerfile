# Use official Python 3.9 Lambda image
FROM public.ecr.aws/lambda/python:3.9

# Set working directory
WORKDIR /var/task

# Copy your predictor code, models, and requirements
COPY predictor.py ./
COPY models ./models
COPY requirements.txt ./

# CRITICAL FIX: Upgrade pip and force reinstall dependencies
# Including "scikit-learn" is crucial as it's the package that actually loads the pickles
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --force-reinstall \
       "numpy==1.23.5" \
       "pandas==1.5.3" \
	"scikit-learn==1.3.0" \
       -r requirements.txt

# Set Lambda handler
CMD ["predictor.lambda_handler"]
