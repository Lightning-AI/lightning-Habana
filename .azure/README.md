# Creation HPU self-hosted agent pool

## Prepare the machine

This is a slightly modified version of the script from
https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/docker

```bash
sudo -i

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    jq \
    git \
    iputils-ping \
    libcurl4 \
    libunwind8 \
    netcat \
    libssl1.0

curl -sL https://aka.ms/InstallAzureCLIDeb | bash
mkdir /azp
sudo chmod 777 /azp
```

## Stating the agents

```bash
export AZP_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxx"
export AZP_URL="https://dev.azure.com/Lightning-AI"
export AZP_POOL="intel-hpus"
export AZP_AGENT_NAME="hpu-1"

git clone https://github.com/Lightning-AI/lightning-Habana.git
cd lightning-Habana
nohup bash .azure/start.sh > "${AZP_AGENT_NAME}.log" &
```

## Check running agents

```bash
ps aux | grep start.sh
```
