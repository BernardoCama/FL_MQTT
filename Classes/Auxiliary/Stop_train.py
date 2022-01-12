import paho.mqtt.subscribe as subscribe
import paho.mqtt.client as mqtt
import _pickle as cPickle
import zlib       
 

def Check_stop_train(params_topic, broker_address, MQTT_port, AUTH_, TLS_):       
        # Receive parameters 
        messages = subscribe.simple(params_topic, qos=2, msg_count=1, retained=True, hostname=broker_address,
                            port=MQTT_port, client_id="Check_stop_train", keepalive=10, will=None, auth=AUTH_, tls=TLS_,
                            protocol=mqtt.MQTTv311)
        body = cPickle.loads(zlib.decompress(messages.payload))

        try:
            if body['TRAIN_STOP']:
                return 1
        except:
            return 0