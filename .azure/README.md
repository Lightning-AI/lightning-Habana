# Creation HPU self-hosted agent pool

## Upgrade machine

In general follow instructions in [Bare Metal Fresh OS Installation](https://docs.habana.ai/en/v1.15.0/Installation_Guide/Bare_Metal_Fresh_OS.html#) and in particular [Habana Driver Unattended Upgrade](https://docs.habana.ai/en/v1.15.0/Installation_Guide/Bare_Metal_Fresh_OS.html#habana-driver-unattended-upgrade).

1. check what is the actual state and version of HW - `hl-smi`
2. check the actual OS version - `lsb_release -a`
3. update sources - `sudo apt update --fix-missing`
4. run upgrade to the latest - `sudo apt upgrade`
5. reboot the machine...

### Troubleshooting

In some cases you may get stack with hanged libs linked to past kernel (for example `sudo apt purge habanalabs-dkms` errors), in such case do not follow with purge old libs and just continue with new

1. try hard removal:
   ```bash
   sudo apt --fix-broken install
   sudo mv /var/lib/dpkg/info /var/lib/dpkg/info_old
   sudo mkdir /var/lib/dpkg/info
   sudo apt-get update && sudo apt-get -f install
   sudo mv /var/lib/dpkg/info/* /var/lib/dpkg/info_old
   sudo rm -rf /var/lib/dpkg/info
   sudo mv /var/lib/dpkg/info_old /var/lib/dpkg/info
   sudo apt-get update && sudo apt-get -f install
   ```
2. purge the hanging package
   ```bash
   apt list --installed | grep habana
   sudo rm /var/lib/dpkg/info/habanalabs-dkms*
   sudo dpkg --configure -D 777 habanalabs-dkms
   sudo apt -f install
   sudo apt purge habanalabs-dkms
   ```
3. if the package folder hangs, drop it:
   ```bash
   sudo rm -rf  /var/lib/dkms/habanalabs-dkms
   ```
4. install all, if some failed try rerun the script
   ```bash
   wget -nv https://vault.habana.ai/artifactory/gaudi-installer/latest/habanalabs-installer.sh
   chmod +x habanalabs-installer.sh
   ./habanalabs-installer.sh install --type base
   ```

Overall if you touch kernel version, follow with reboot before any eventual other updates.

## Try running a docker container

This si simple validation that all is working as expected and via Docker we can correctly access the HW.

```bash
sudo docker run \
   --runtime=habana \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   --cap-add=sys_nice \
   --ipc=host \
   vault.habana.ai/gaudi-docker/1.15.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest \
   hl-smi
```

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
