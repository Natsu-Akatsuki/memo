#!/bin/bash

# 1.填写有线网络的connection name
connection_name=helios

while true; do
    read -r -p "set static IP for lidar? [y/n]" input
    case $input in
        [yY][eE][sS] | [yY])
            # 2.填写有线网络的地址和网关
            nmcli connection modify ${connection_name} \
                ipv4.method manual \
                ipv4.addresses 192.168.0.110/16 \
                ipv4.gateway 192.168.0.1
            echo "set static IP"
            break
            ;;

        [nN][oO] | [nN])
            nmcli connection modify ${connection_name} ipv4.method auto
            echo "set dynamic IP"
            echo -e "\n!!!   ATTENTION: if the bash blocked, it may be:"
            echo '!!!   "Error: Connection activation failed: IP configuration could not be reserved (no available address, timeout, etc.)"'
            echo -e "!!!   then you can exit(i.g. press ctrl+c) the bash and check the link\n"
            break
            ;;

        *)
            echo "Invalid input..."
            ;;
    esac
done

nmcli connection up ${connection_name}
echo "modify the IP successfully!"
