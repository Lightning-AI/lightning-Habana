# Creation HPU self-hosted agent pool

## Upgrade machine

In general follow instructions in [Bare Metal Fresh OS Installation](https://docs.habana.ai/en/v1.10.0/Installation_Guide/Bare_Metal_Fresh_OS.html#) and in particular [Habana Driver Unattended Upgrade](https://docs.habana.ai/en/v1.10.0/Installation_Guide/Bare_Metal_Fresh_OS.html#habana-driver-unattended-upgrade).

1. check what is the actual state and version of HW - `hl-smi`
1. check the actual OS version - `lsb_release -a`
1. update sources - `sudo apt update --fix-missing`
1. run upgrade to the latest - `sudo apt upgrade`
1. reboot the machine...

### Troubleshooting

In some cases you may get stack with hanged libs linked to past kernel (for example `sudo apt purge habanalabs-dkms` errors), in such case do not follow with purge old libs and just continue with new

1. list available libs: `sudo apt list | grep habana | grep 1.12`
1. install all libs from the output list above:
   ```bash
   sudo apt install \
     habanalabs-container-runtime \
     habanalabs-dkms \
     habanalabs-firmware-tools \
     habanalabs-firmware \
     habanalabs-graph \
     habanalabs-qual \
     habanalabs-rdma-core \
     habanalabs-thunk \
     habanatools
   ```

Overall if you touch kernel version, follow with reboot before any eventual other updates.

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
git clone https://github.com/Lightning-AI/lightning-Habana.git
cd lightning-Habana

export AZP_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxx"
export AZP_URL="https://dev.azure.com/Lightning-AI"
export AZP_POOL="intel-hpus"

for i in {0..7..2}
do
     nohup bash .azure/start.sh "AZP_AGENT_NAME=HPU_$i,$((i+1))" > "agent-$i.log" &
done
```

## Check running agents

```bash
ps aux | grep start.sh
```
