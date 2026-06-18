#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-nba-analytics}"
AWS_REGION="${AWS_REGION:-us-west-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.small}"
KEY_NAME="${KEY_NAME:-${APP_NAME}-key}"
SECURITY_GROUP_NAME="${SECURITY_GROUP_NAME:-${APP_NAME}-sg}"
ALLOWED_CIDR="${ALLOWED_CIDR:-$(curl -s https://checkip.amazonaws.com)/32}"
LOCAL_KEY_PATH="${LOCAL_KEY_PATH:-$HOME/.ssh/${KEY_NAME}.pem}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/nbaAnalytics}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_DATA_FILE="${ROOT_DIR}/deploy/ec2_user_data.sh"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command aws
require_command curl
require_command rsync
require_command ssh

echo "Using AWS region: ${AWS_REGION}"
aws sts get-caller-identity >/dev/null

VPC_ID="$(aws ec2 describe-vpcs \
  --region "${AWS_REGION}" \
  --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' \
  --output text)"

if [[ "${VPC_ID}" == "None" || -z "${VPC_ID}" ]]; then
  echo "No default VPC found in ${AWS_REGION}." >&2
  exit 1
fi

SUBNET_ID="$(aws ec2 describe-subnets \
  --region "${AWS_REGION}" \
  --filters Name=vpc-id,Values="${VPC_ID}" Name=default-for-az,Values=true \
  --query 'Subnets[0].SubnetId' \
  --output text)"

SECURITY_GROUP_ID="$(aws ec2 describe-security-groups \
  --region "${AWS_REGION}" \
  --filters Name=group-name,Values="${SECURITY_GROUP_NAME}" Name=vpc-id,Values="${VPC_ID}" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)"

if [[ "${SECURITY_GROUP_ID}" == "None" || -z "${SECURITY_GROUP_ID}" ]]; then
  SECURITY_GROUP_ID="$(aws ec2 create-security-group \
    --region "${AWS_REGION}" \
    --group-name "${SECURITY_GROUP_NAME}" \
    --description "NBA analytics API and dashboard" \
    --vpc-id "${VPC_ID}" \
    --query 'GroupId' \
    --output text)"
fi

for PORT in 22 8000 8501; do
  aws ec2 authorize-security-group-ingress \
    --region "${AWS_REGION}" \
    --group-id "${SECURITY_GROUP_ID}" \
    --protocol tcp \
    --port "${PORT}" \
    --cidr "${ALLOWED_CIDR}" >/dev/null 2>&1 || true
done

if [[ ! -f "${LOCAL_KEY_PATH}" ]]; then
  mkdir -p "$(dirname "${LOCAL_KEY_PATH}")"
  aws ec2 create-key-pair \
    --region "${AWS_REGION}" \
    --key-name "${KEY_NAME}" \
    --query 'KeyMaterial' \
    --output text > "${LOCAL_KEY_PATH}"
  chmod 600 "${LOCAL_KEY_PATH}"
fi

AMI_ID="$(aws ssm get-parameter \
  --region "${AWS_REGION}" \
  --name /aws/service/canonical/ubuntu/server/22.04/stable/current/arm64/hvm/ebs-gp2/ami-id \
  --query 'Parameter.Value' \
  --output text)"

INSTANCE_ID="$(aws ec2 run-instances \
  --region "${AWS_REGION}" \
  --image-id "${AMI_ID}" \
  --instance-type "${INSTANCE_TYPE}" \
  --key-name "${KEY_NAME}" \
  --security-group-ids "${SECURITY_GROUP_ID}" \
  --subnet-id "${SUBNET_ID}" \
  --associate-public-ip-address \
  --user-data "file://${USER_DATA_FILE}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP_NAME}}]" \
  --query 'Instances[0].InstanceId' \
  --output text)"

echo "Launched instance: ${INSTANCE_ID}"
aws ec2 wait instance-status-ok --region "${AWS_REGION}" --instance-ids "${INSTANCE_ID}"

PUBLIC_IP="$(aws ec2 describe-instances \
  --region "${AWS_REGION}" \
  --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"

echo "Instance public IP: ${PUBLIC_IP}"
echo "Waiting for SSH..."
until ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "${LOCAL_KEY_PATH}" "ubuntu@${PUBLIC_IP}" "echo ok" >/dev/null 2>&1; do
  sleep 10
done

rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  -e "ssh -o StrictHostKeyChecking=no -i ${LOCAL_KEY_PATH}" \
  "${ROOT_DIR}/" "ubuntu@${PUBLIC_IP}:${REMOTE_DIR}/"

ssh -o StrictHostKeyChecking=no -i "${LOCAL_KEY_PATH}" "ubuntu@${PUBLIC_IP}" \
  "cd ${REMOTE_DIR} && docker compose up -d --build db api dashboard"

echo "API health:    http://${PUBLIC_IP}:8000/health"
echo "Dashboard:     http://${PUBLIC_IP}:8501"
echo "Benchmark:"
echo "  python src/benchmark_api.py --url http://${PUBLIC_IP}:8000 --requests 1000 --warmup 50"
echo "Terminate when done:"
echo "  aws ec2 terminate-instances --region ${AWS_REGION} --instance-ids ${INSTANCE_ID}"
