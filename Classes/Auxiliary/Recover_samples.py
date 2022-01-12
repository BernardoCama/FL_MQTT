import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import _pickle as cPickle
import zlib       
 

def Recover_samples(client_, num_images_train, num_images_test, num_images_valid, NUM_CLIENTS, samples_topic, broker_address, MQTT_port, AUTH_, TLS_):    

        def on_connect(client, userdata, flags, rc):
            client.subscribe(samples_topic + '/#', qos=2)

        def on_message(client, userdata, messages):

            for message in (messages if type(messages) is list else [messages]):

                body = cPickle.loads(zlib.decompress(message.payload))
                client_ = body['client']
                num_images_train[client_] = body['num_images_train']          
                num_images_test[client_] = body['num_images_test']          
                num_images_valid[client_] = body['num_images_valid']  


        if type(client_) == int:

            # Share number of samples
            messages = []
            payload = zlib.compress(cPickle.dumps({'client':client_,
                                                    'num_images_train':num_images_train[client_],
                                                    'num_images_test':num_images_test[client_],
                                                    'num_images_valid':num_images_valid[client_]
            }))
            messages.append({'topic':samples_topic + '/{}'.format(client_), 'payload': payload, 'qos': 2, 'retain': True})
            publish.multiple(messages, hostname=broker_address, port=MQTT_port, client_id="Client{}".format(client_), keepalive=10,
                            will=None, auth=AUTH_, tls=TLS_, protocol=mqtt.MQTTv311, transport="tcp")

        else:

            client = mqtt.Client(client_id="Receive_Samples", clean_session=None, userdata=None, protocol=mqtt.MQTTv311, transport="tcp")
            if AUTH_ is not None:
                client.username_pw_set(AUTH_["username"], AUTH_["password"])
                client.tls_set(ca_certs=TLS_["ca_certs"], certfile=None, keyfile=None, cert_reqs=mqtt.ssl.CERT_REQUIRED, tls_version=TLS_["tls_version"], ciphers=None)
            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(broker_address, port=MQTT_port, keepalive=1000)
            client.loop_start()                                   
