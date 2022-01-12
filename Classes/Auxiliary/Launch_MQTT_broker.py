from Classes.Params import param

def Launch_MQTT_broker():

    if param.LAUNCH_MQTT_BROKER:
        if not param.REMOTE:
            # Inizialize MQTT server
            if param.OS == 'MACOS':
                param.appscript.app('Terminal').do_script('killall mosquitto; exit')
            elif param.OS == 'UBUNTU':
                param.os.system("gnome-terminal -e 'bash -c \"killall mosquitto; exec bash\"'")
            elif param.OS == 'WINDOWS':
                param.os.system('start taskkill /IM mosquitto.exe')
            try:
                param.time.sleep(3)
                if param.OS == 'MACOS':
                    param.appscript.app('Terminal').do_script('/usr/local/sbin/mosquitto -p ' + str(param.MQTT_port) + '; exit') 
                elif param.OS == 'UBUNTU':    
                    param.os.system("gnome-terminal -e 'bash -c \"mosquitto -c \"/etc/mosquitto/conf.d/mosquitto.conf\"; exec bash\"'")
                elif param.OS == 'WINDOWS':
                    param.os.system('start  ""  ' + param.MQTT_broker_dir + ' -c ' + param.MQTT_broker_config_file + ' -v')
                param.time.sleep(3)
            finally:
                pass
