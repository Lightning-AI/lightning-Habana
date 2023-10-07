# Creation HPU self-hosted agent pool

## Upgrade machine

In general follow instructions in [Bare Metal Fresh OS Installation](https://docs.habana.ai/en/v1.10.0/Installation_Guide/Bare_Metal_Fresh_OS.html#) and in particular [Habana Driver Unattended Upgrade](https://docs.habana.ai/en/v1.10.0/Installation_Guide/Bare_Metal_Fresh_OS.html#habana-driver-unattended-upgrade).

1. check what is the actual state and version of HW - `hl-smi`
2. check the actual OS version - `lsb_release -a`
3. run upgrade to the latest - `sudo apt install --only-upgrade habanalabs-dkms`
4. reboot the machine...

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
