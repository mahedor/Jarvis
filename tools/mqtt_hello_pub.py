"""
MQTT hello-world publisher - jarvis/test/hello
=================================================
THROWAWAY DIAGNOSTIC - not production code, and deliberately not wired into
anything. The sending half of the pair; see tools/mqtt_hello_sub.py for the
receiving half and the README ("MQTT hello-world") for the run order.

Scenarios:
    qos       one QoS 0 and one QoS 1 message, sent while the subscriber is
              offline. Only the QoS 1 one should survive.
    retained  a single retained message, so a subscriber that starts LATER
              still learns the current state.
    will      registers a last-will with the broker, connects, then dies
              without a clean DISCONNECT so the broker publishes the will.
    clear     wipes the retained message (empty payload, retain=True) so the
              topic does not leak state into the next demo.

Usage:
    python tools/mqtt_hello_pub.py --scenario qos
    python tools/mqtt_hello_pub.py --scenario retained
    python tools/mqtt_hello_pub.py --scenario will --die-after 4
    python tools/mqtt_hello_pub.py --scenario clear

Assumes an anonymous broker on 127.0.0.1:1883 (the mosquitto 2.x default).
"""

import argparse
import json
import os
import time

import paho.mqtt.client as mqtt

TOPIC = "jarvis/test/hello"


def stamp():
    return time.strftime("%H:%M:%S")


def payload(note, **extra):
    body = {"note": note, "sent_at": stamp()}
    body.update(extra)
    return json.dumps(body)


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code != 0:
        print(f"[{stamp()}] CONNECT FAILED: {reason_code}", flush=True)
    else:
        print(f"[{stamp()}] connected", flush=True)


def send(client, body, qos, retain=False):
    info = client.publish(TOPIC, body, qos=qos, retain=retain)
    # For QoS 1 this blocks until the broker's PUBACK lands, which is the whole
    # difference being demonstrated. For QoS 0 it returns as soon as the bytes
    # hit the socket - nobody ever confirms them.
    info.wait_for_publish(timeout=5)
    print(
        f"[{stamp()}] PUBLISHED  qos={qos} retain={retain} "
        f"acked={info.is_published()}\n"
        f"                payload={body}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["qos", "retained", "will", "clear"], required=True)
    parser.add_argument(
        "--die-after",
        type=float,
        default=None,
        help="will scenario: hard-exit after N seconds (no DISCONNECT packet), "
        "which is what a kill -9 looks like to the broker. Omit to sit until "
        "you kill it yourself.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    print(f"=== publisher | scenario={args.scenario} ===", flush=True)

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"jarvis-hello-pub-{int(time.time())}",
        protocol=mqtt.MQTTv311,
    )
    client.on_connect = on_connect

    if args.scenario == "will":
        # Must be set BEFORE connect - the will travels inside the CONNECT
        # packet and the broker holds it for the life of the connection.
        will = payload("publisher died unexpectedly", event="last_will", present=False)
        client.will_set(TOPIC, will, qos=1, retain=False)
        print(f"[{stamp()}] will registered: {will}", flush=True)

    client.connect(args.host, args.port, keepalive=10)
    client.loop_start()
    time.sleep(0.5)  # let CONNACK land before we start talking

    if args.scenario == "qos":
        send(client, payload("QoS 0 - fire and forget", qos=0), qos=0)
        send(client, payload("QoS 1 - at least once", qos=1), qos=1)

    elif args.scenario == "retained":
        send(client, payload("michael is home", present=True), qos=1, retain=True)

    elif args.scenario == "clear":
        client.publish(TOPIC, "", qos=1, retain=True).wait_for_publish(timeout=5)
        print(f"[{stamp()}] retained message cleared", flush=True)

    elif args.scenario == "will":
        send(client, payload("publisher online", event="hello", present=True), qos=1)
        if args.die_after is None:
            print(f"[{stamp()}] alive - kill this process to fire the will", flush=True)
            while True:
                time.sleep(1)
        print(f"[{stamp()}] dying hard in {args.die_after:.0f}s (no DISCONNECT)...", flush=True)
        time.sleep(args.die_after)
        print(f"[{stamp()}] *** os._exit - socket dies, broker fires the will ***", flush=True)
        os._exit(1)

    client.loop_stop()
    client.disconnect()
    print(f"[{stamp()}] clean disconnect", flush=True)


if __name__ == "__main__":
    main()
