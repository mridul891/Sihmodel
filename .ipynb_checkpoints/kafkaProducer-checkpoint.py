import pandas as pd 
from kafka import KafkaConsumer , KafkaProducer
from time import sleep 
from json import dumps 
import json

producer = KafkaProducer(bootstrap_servers= ['13.233.91.150:9092'],
                        value_serializer = lambda x:
                         dumps(x).encode('utf-8'))
producer.send('demoTest' , value={'mridulasdfaf ' : 'pandeys'})