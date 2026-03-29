#!/bin/bash

INSTANCE_ID=$(aws ec2 run-instances \
--image-id ami-05d2d839d4f73aafb \
--count 1 \
--instance-type t2.micro \
--key-name postershaala \
--security-group-ids sg-08202a87a1b0a41c5 \
--query 'Instances[0].InstanceId' \
--output text)

echo "Instance launched: $INSTANCE_ID"


IP=$(aws ec2 describe-instances \
--instance-ids $INSTANCE_ID \
--query 'Reservations[0].Instances[0].PublicIpAddress' \
--output text)

ssh -i postershaala.pem -o StrictHostKeyChecking=no ubuntu@$IP << EOF
sudo apt update
sudo apt install docker.io -y
git clone  https://github.com/manvendrapratapsinghdev/IITJMaterial.git
cd IITJMaterial/Sem-3/VCC/Assignments/3/auto-scale
sudo docker build -t autoscale .
sudo docker run -d -p 80:5000 autoscale:latest
EOF