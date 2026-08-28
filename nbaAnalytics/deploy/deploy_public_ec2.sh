#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-nba-courtvision}"
AWS_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t4g.small}"
KEY_NAME="${KEY_NAME:-${APP_NAME}-key}"
SECURITY_GROUP_NAME="${SECURITY_GROUP_NAME:-${APP_NAME}-web}"
LOCAL_KEY_PATH="${LOCAL_KEY_PATH:-${HOME}/.ssh/${KEY_NAME}.pem}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/nba-courtvision}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_DATA_FILE="${ROOT_DIR}/deploy/ec2_user_data.sh"

for command in aws curl rsync ssh; do
  command -v "${command}" >/dev/null 2>&1 || { echo "Missing required command: ${command}" >&2; exit 1; }
done

aws sts get-caller-identity >/dev/null
CLIENT_CIDR="${ALLOWED_SSH_CIDR:-$(curl -fsS https://checkip.amazonaws.com)/32}"

VPC_ID="$(aws ec2 describe-vpcs --region "${AWS_REGION}" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
if [[ -z "${VPC_ID}" || "${VPC_ID}" == "None" ]]; then
  echo "No default VPC exists in ${AWS_REGION}." >&2
  exit 1
fi

SUBNET_ID="$(aws ec2 describe-subnets --region "${AWS_REGION}" --filters Name=vpc-id,Values="${VPC_ID}" Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text)"
SECURITY_GROUP_ID="$(aws ec2 describe-security-groups --region "${AWS_REGION}" --filters Name=group-name,Values="${SECURITY_GROUP_NAME}" Name=vpc-id,Values="${VPC_ID}" --query 'SecurityGroups[0].GroupId' --output text)"
if [[ -z "${SECURITY_GROUP_ID}" || "${SECURITY_GROUP_ID}" == "None" ]]; then
  SECURITY_GROUP_ID="$(aws ec2 create-security-group --region "${AWS_REGION}" --group-name "${SECURITY_GROUP_NAME}" --description "Public NBA CourtVision dashboard" --vpc-id "${VPC_ID}" --query GroupId --output text)"
fi

aws ec2 authorize-security-group-ingress --region "${AWS_REGION}" --group-id "${SECURITY_GROUP_ID}" --protocol tcp --port 80 --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
aws ec2 authorize-security-group-ingress --region "${AWS_REGION}" --group-id "${SECURITY_GROUP_ID}" --protocol tcp --port 22 --cidr "${CLIENT_CIDR}" >/dev/null 2>&1 || true

if [[ ! -f "${LOCAL_KEY_PATH}" ]]; then
  mkdir -p "$(dirname "${LOCAL_KEY_PATH}")"
  aws ec2 create-key-pair --region "${AWS_REGION}" --key-name "${KEY_NAME}" --query KeyMaterial --output text > "${LOCAL_KEY_PATH}"
  chmod 600 "${LOCAL_KEY_PATH}"
fi

INSTANCE_ID="$(aws ec2 describe-instances --region "${AWS_REGION}" --filters Name=tag:Name,Values="${APP_NAME}" Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[0].Instances[0].InstanceId' --output text)"
if [[ -z "${INSTANCE_ID}" || "${INSTANCE_ID}" == "None" ]]; then
  AMI_ID="$(aws ssm get-parameter --region "${AWS_REGION}" --name /aws/service/canonical/ubuntu/server/22.04/stable/current/arm64/hvm/ebs-gp2/ami-id --query Parameter.Value --output text)"
  INSTANCE_ID="$(aws ec2 run-instances --region "${AWS_REGION}" --image-id "${AMI_ID}" --instance-type "${INSTANCE_TYPE}" --key-name "${KEY_NAME}" --security-group-ids "${SECURITY_GROUP_ID}" --subnet-id "${SUBNET_ID}" --associate-public-ip-address --user-data "file://${USER_DATA_FILE}" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP_NAME}},{Key=Application,Value=NBA-CourtVision}]" --query 'Instances[0].InstanceId' --output text)"
else
  STATE="$(aws ec2 describe-instances --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}" --query 'Reservations[0].Instances[0].State.Name' --output text)"
  if [[ "${STATE}" == "stopped" ]]; then aws ec2 start-instances --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}" >/dev/null; fi
fi

echo "Waiting for ${INSTANCE_ID}..."
aws ec2 wait instance-status-ok --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"
PUBLIC_IP="$(aws ec2 describe-instances --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

until ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "${LOCAL_KEY_PATH}" "ubuntu@${PUBLIC_IP}" "docker --version" >/dev/null 2>&1; do sleep 5; done
ssh -o StrictHostKeyChecking=no -i "${LOCAL_KEY_PATH}" "ubuntu@${PUBLIC_IP}" "mkdir -p ${REMOTE_DIR}/hf_space ${REMOTE_DIR}/deploy"
rsync -az --delete --exclude __pycache__ --exclude .DS_Store -e "ssh -o StrictHostKeyChecking=no -i ${LOCAL_KEY_PATH}" "${ROOT_DIR}/hf_space/" "ubuntu@${PUBLIC_IP}:${REMOTE_DIR}/hf_space/"
rsync -az -e "ssh -o StrictHostKeyChecking=no -i ${LOCAL_KEY_PATH}" "${ROOT_DIR}/deploy/docker-compose.public.yml" "ubuntu@${PUBLIC_IP}:${REMOTE_DIR}/deploy/"
ssh -o StrictHostKeyChecking=no -i "${LOCAL_KEY_PATH}" "ubuntu@${PUBLIC_IP}" "cd ${REMOTE_DIR}/deploy && docker compose -f docker-compose.public.yml up -d --build"

for attempt in {1..24}; do
  if curl -fsS "http://${PUBLIC_IP}/_stcore/health" >/dev/null; then
    echo "Dashboard: http://${PUBLIC_IP}"
    echo "Instance:  ${INSTANCE_ID} (${AWS_REGION})"
    exit 0
  fi
  sleep 5
done
echo "Deployment completed, but the health check did not become ready: http://${PUBLIC_IP}" >&2
exit 1
