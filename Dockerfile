# 1. Start with the exact same Spark base image we verified works
FROM bitnamilegacy/spark:3.5.1

# (Bitnami images run as a restricted user by default for security)
USER root

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy ONLY the requirements file first
COPY requirement.txt .

# 5. Install the required Python packages directly into the image
RUN pip install --no-cache-dir -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 6. Copy the rest of your project files into the container's /app folder
COPY . .

# 7. Switch back to the safe, non-root user (User ID 1001 for Bitnami) before running
USER 1001
