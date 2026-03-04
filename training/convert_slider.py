from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, get_types_from_msg, Stores
import numpy as np
from pathlib import Path

# Manually define the custom message types — copied exactly from the warnings above
CUSTOM_MSGS = {
    'dv_ros_msgs/msg/Event': """
uint16 x
uint16 y
time ts
bool polarity
""",
    'dv_ros_msgs/msg/EventArray': """
Header header
uint32 height
uint32 width
dv_ros_msgs/Event[] events
""",
    'strttc_msgs/msg/ttc_message': """
std_msgs/Time stamp
std_msgs/Float64 ttc
""",
    'strttc_msgs/msg/boundingbox_message': """
std_msgs/Time stamp
int64 xmin
int64 ymin
int64 xmax
int64 ymax
""",
}

def build_typestore():
    typestore = get_typestore(Stores.ROS1_NOETIC)
    add_types = {}
    for typename, msgdef in CUSTOM_MSGS.items():
        add_types.update(get_types_from_msg(msgdef.strip(), typename))
    typestore.register(add_types)
    return typestore


def extract_slider(bag_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    typestore = build_typestore()

    all_t, all_x, all_y, all_p = [], [], [], []
    ttc_times, ttc_values = [], []

    with Reader(bag_path) as reader:
        # Match by message type, not topic name — works across all three bags
        event_conns = [c for c in reader.connections
                       if 'EventArray' in c.msgtype]
        ttc_conns   = [c for c in reader.connections
                       if 'ttc_message' in c.msgtype]

        print(f"Using event topic:  {[c.topic for c in event_conns]}")
        print(f"Using TTC topic:    {[c.topic for c in ttc_conns]}")

        print(f"Reading events from {bag_path}...")
        for conn, timestamp, rawdata in reader.messages(connections=event_conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            for event in msg.events:
                t_us = int(event.ts.sec) * 1_000_000 + int(event.ts.nanosec) // 1000
                all_t.append(t_us)
                all_x.append(int(event.x))
                all_y.append(int(event.y))
                all_p.append(int(event.polarity))

        print(f"Reading TTC from {bag_path}...")
        for conn, timestamp, rawdata in reader.messages(connections=ttc_conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            t_us = int(msg.stamp.data.sec) * 1_000_000 + int(msg.stamp.data.nanosec) // 1000
            ttc_values.append(float(msg.ttc.data))
            ttc_times.append(t_us)

    if not all_t:
        print("ERROR: No events extracted!")
        return

    np.savez_compressed(str(output_dir / 'events.npz'),
        t=np.array(all_t,    dtype=np.int64),
        x=np.array(all_x,    dtype=np.uint16),
        y=np.array(all_y,    dtype=np.uint16),
        p=np.array(all_p,    dtype=np.int8),
    )
    np.savez_compressed(str(output_dir / 'ttc_gt.npz'),
        t=np.array(ttc_times,  dtype=np.int64),
        ttc=np.array(ttc_values, dtype=np.float32),
    )
    print(f"Done — Events: {len(all_t):,} | TTC msgs: {len(ttc_values)} | "
          f"TTC range: [{min(ttc_values):.2f}, {max(ttc_values):.2f}]s")



extract_slider('data/slider/slider_500/slider_500.bag',   'data/slider/Slider500')
extract_slider('data/slider/slider_750/slider_750.bag',   'data/slider/Slider750')
extract_slider('data/slider/slider_1000/slider_1000.bag', 'data/slider/Slider1000')
