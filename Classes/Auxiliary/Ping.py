import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import _pickle as cPickle
import zlib  
import time    
 

def Ping(client_, num_clients, Working_clients, Last_seen_time_topic, broker_address, MQTT_port, AUTH_, TLS_, Stop_Training = None):   

        global NUM_CLIENTS

        def on_connect(client, userdata, flags, rc):
            client.subscribe(Last_seen_time_topic + '/#', qos=2)

        def on_message(client, userdata, messages):
            global NUM_CLIENTS
            now = time.time()
            old_number_of_clients = len(Working_clients)

            for message in (messages if type(messages) is list else [messages]):

                body = cPickle.loads(zlib.decompress(message.payload))
                client_ = body['client']
                time_ = body['time']

                # Remove from Clients
                if int(now - time_) >= 60 * 30:
                    Working_clients.pop(client_, None)
                    # print('Client {} not active\n'.format(client_))

                # Insert the Client
                else:
                    Working_clients[client_] = client_
                    # print('Client {} active\n'.format(client_))

            NUM_CLIENTS = len(Working_clients)
            if NUM_CLIENTS != old_number_of_clients:
                print('Active Clients: {}'.format(Working_clients.keys()))


        if type(client_) == int:

            if Stop_Training is not None:
                if Stop_Training:
                    # Notify Stop train
                    messages = []
                    payload = zlib.compress(cPickle.dumps({'client':client_,
                                                            'time': 0,
                    }))
                
            else:
                # Recover last time Client seen
                messages = []
                payload = zlib.compress(cPickle.dumps({'client':client_,
                                                        'time': time.time(),
                }))

            messages.append({'topic':Last_seen_time_topic + '/{}'.format(client_), 'payload': payload, 'qos': 2, 'retain': True})
            publish.multiple(messages, hostname=broker_address, port=MQTT_port, client_id="Client{}".format(client_), keepalive=1000,
                            will=None, auth=AUTH_, tls=TLS_, protocol=mqtt.MQTTv311, transport="tcp")

        else:

            print("Listening for Clients\n")
            client = mqtt.Client(client_id="Listening_Ping", clean_session=None, userdata=None, protocol=mqtt.MQTTv311, transport="tcp")
            if AUTH_ is not None:
                client.username_pw_set(AUTH_["username"], AUTH_["password"])
                client.tls_set(ca_certs=TLS_["ca_certs"], certfile=None, keyfile=None, cert_reqs=mqtt.ssl.CERT_REQUIRED, tls_version=TLS_["tls_version"], ciphers=None)
            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(broker_address, port=MQTT_port, keepalive=1000)
            client.loop_start()
            
        return 
                                