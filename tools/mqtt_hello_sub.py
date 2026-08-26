"""
MQTT hello-world subscriber - jarvis/presence/test
==================================================
THROWAWAY DIAGNOSTIC - not production code, and deliberately not wired into
anything. This is the receiving half of a two-script pair used to prove out the
MQTT semantics the presence service will depend on before any of it is written:

    QoS      does a message survive the subscriber being offline?
    RETAIN   does a late subscriber learn the current state?
    WILL     does the broker announce it when a publisher dies?

Pair it with tools/mqtt_hello_pub.py. Every scenario below has matching notes in
the README under "MQTT hello-world".

Usage:
    python tools/mqtt_hello_sub.py --scenario qos --phase register
    python tools/mqtt_hello_sub.py --scenario qos --phase collect
    python tools/mqtt_hello_sub.py --scenario retained
    python tools/mqtt_hello_sub.py --scenario will --seconds 15
    python tools/mqtt_hello_sub.py --scenario wipe        # tidy up afterwards

Assumes an anonymous broker on 127.0.0.1:1883 (the mosquitto 2.x default).
"""

import argparse
import time

import paho.mqtt.client as mqtt

TOPIC = "jarvis/presence/test"

# The QoS demo needs the broker to hold a session for us while we are offline,
# and a session is keyed by client id - so that one scenario gets a FIXED id.
# The other two want a clean slate every run and use a throwaway id.
PERSISTENT_CLIENT_ID = "jarvis-hello-sub-persistent"


def stamp():
    return time.strftime("%H:%M:%S")


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code != 0:
        print(f"[{stamp()}] CONNECT FAILED: {reason_code}", flush=True)
        return
    # session_present is the broker telling us "I still had your subscriptions
    # and your queued messages". False means it created a fresh session.
    print(
        f"[{stamp()}] connected  (session_present={flags.session_present})",
        flush=True,
    )
    if userdata["subscribe"]:
        client.subscribe(TOPIC, qos=userdata["qos"])
        print(f"[{stamp()}] subscribing to {TOPIC} at QoS {userdata['qos']}", flush=True)


def on_subscribe(client, userdata, mid, reason_code_list, properties):
    granted = ", ".join(str(rc.value) for rc in reason_code_list)
    print(f"[{stamp()}] SUBACK - broker granted QoS {granted}", flush=True)
    userdata["subscribed"] = True


def on_message(client, userdata, msg):
    userdata["received"].append(msg)
    print(
        f"[{stamp()}] >>> MESSAGE  topic={msg.topic}  qos={msg.qos}  "
        f"retain={msg.retain}\n"
        f"                payload={msg.payload.decode()}",
        flush=True,
    )


def build_client(client_id, clean_session, userdata):
    # MQTTv311 on purpose: clean_session=False is the v3.1.1 spelling of a
    # persistent session, and it is the simplest way to show queueing. (v5 says
    # the same thing with clean_start + a session-expiry property.)
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        clean_session=clean_session,
        protocol=mqtt.MQTTv311,
        userdata=userdata,
    )
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["qos", "retained", "will", "wipe"], required=True)
    parser.add_argument(
        "--phase",
        choices=["register", "collect"],
        default="collect",
        help="qos scenario only: 'register' claims the persistent session and "
        "exits; 'collect' reconnects to it and drains whatever queued up.",
    )
    parser.add_argument("--seconds", type=float, default=8.0, help="how long to listen")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    if args.scenario == "wipe":
        # Housekeeping. The persistent session from the QoS demo keeps queueing
        # QoS 1 messages for a subscriber that never comes back. Connecting with
        # the same id but clean_session=True tells the broker to throw it away.
        wiper = build_client(PERSISTENT_CLIENT_ID, True, {"received": [], "subscribe": False, "qos": 1})
        wiper.connect(args.host, args.port, keepalive=10)
        wiper.loop_start()
        time.sleep(0.5)
        wiper.loop_stop()
        wiper.disconnect()
        print(f"[{stamp()}] persistent session '{PERSISTENT_CLIENT_ID}' wiped", flush=True)
        return

    if args.scenario == "qos":
        client_id = PERSISTENT_CLIENT_ID
        clean_session = False
        # Only subscribe during 'register'. The whole point is that the
        # subscription outlives the disconnect, so 'collect' must NOT re-send it.
        subscribe = args.phase == "register"
    else:
        client_id = f"jarvis-hello-sub-{int(time.time())}"
        clean_session = True
        subscribe = True

    userdata = {"received": [], "subscribe": subscribe, "subscribed": False, "qos": 1}

    print(f"=== subscriber | scenario={args.scenario} phase={args.phase} ===", flush=True)
    print(f"    client_id={client_id}  clean_session={clean_session}", flush=True)

    client = build_client(client_id, clean_session, userdata)
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()

    if args.scenario == "qos" and args.phase == "register":
        # Wait for the SUBACK, then leave. From here on we are "offline".
        deadline = time.time() + 5
        while not userdata["subscribed"] and time.time() < deadline:
            time.sleep(0.05)
        client.loop_stop()
        client.disconnect()
        print(f"[{stamp()}] registered, disconnecting - subscriber is now OFFLINE", flush=True)
        return

    print(f"[{stamp()}] listening for {args.seconds:.0f}s...", flush=True)
    time.sleep(args.seconds)
    client.loop_stop()
    client.disconnect()

    print(f"--- received {len(userdata['received'])} message(s) ---", flush=True)
    for msg in userdata["received"]:
        print(f"    qos={msg.qos} retain={msg.retain} {msg.payload.decode()}", flush=True)


if __name__ == "__main__":
    main()
