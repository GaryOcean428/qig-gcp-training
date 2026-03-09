FROM nvcr.io/nvidia/pytorch:24.01-py3

# Core dependencies
RUN pip install \
    scipy \
        tensorboard \
            google-cloud-storage \
                pyyaml

                # QIG-specific code
                COPY qig_kernel/ /app/qig_kernel/
                COPY training/ /app/training/
                COPY coordizer/ /app/coordizer/

                WORKDIR /app
                ENTRYPOINT ["python", "-m", "training.train"]
                
