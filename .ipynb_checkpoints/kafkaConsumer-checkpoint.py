from kafka import KafkaConsumer , KafkaProducer
from time import sleep 
from json import dumps ,loads
import json

consumer = KafkaConsumer(
    'demoTest',
    bootstrap_servers= ['13.233.91.150:9092'],
    value_deserializer=lambda x: loads(x.decode('utf-8')))


for c in consumer:
    print(c.value)