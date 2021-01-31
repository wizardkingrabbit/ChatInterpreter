import json
import numpy

# load data
file_path = input("chats that you wish to process ----> ")
with open(file_path, 'rb') as f:
    data = f.read()
    jsondata = json.loads(data)

# basic information
chats_length = len(jsondata["comments"])
stream_length = jsondata["video"]["end"] - jsondata["video"]["start"]
average_space = stream_length / chats_length
print("there are " + str(chats_length) + " comments")
print("the length of this stream is " + str(stream_length))
print("the average space between chats is " + str(average_space))

# checkpoint
while(True):
    user_cmd = input("Give a valid index to check the corresponding chat, or enter q to advance to the next part ----> ")
    if (user_cmd == 'q'):
        break
    else:
        print("time: " + str(jsondata["comments"][int(user_cmd)]["content_offset_seconds"]))
        chat_content = ""
        for word in jsondata["comments"][int(user_cmd)]["message"]["fragments"]:
            chat_content += word["text"] + " "
        print("content: " + chat_content)

# the linear clustering process
# step 1: tools
print("the clustring process starts")
timestamp_array = numpy.zeros(chats_length)
counter = 0
while (counter < chats_length):
    timestamp_array[counter] = jsondata["comments"][counter]["content_offset_seconds"]
    counter += 1
expected_cluster = stream_length / 60 # I personally expect that there should be one cluster every minute on average
space_thresh = average_space # max space that two chats in the same cluster can have
chats_thresh = chats_length / expected_cluster # minimal number of chats that required to form a cluster
cluster_array = numpy.zeros([int(expected_cluster), 6])
# 0 --> starting chat of the cluster
# 1 --> ending chat of the cluster
# 2 --> starting time of the cluster
# 3 --> ending time of the cluster
# 4 --> cluster size
# 5 --> cluster length

# step 2: iteration
potential_start = 0
cluster_counter = 0
counter = 1
while (counter < chats_length):
    if (timestamp_array[counter] - timestamp_array[counter - 1] <= space_thresh):
        pass # go to the next chat
    elif (counter - potential_start >= chats_thresh):
        cluster_array[cluster_counter][0] = potential_start
        cluster_array[cluster_counter][1] = counter
        cluster_array[cluster_counter][2] = timestamp_array[potential_start]
        cluster_array[cluster_counter][3] = timestamp_array[counter]
        cluster_array[cluster_counter][4] = counter - potential_start
        cluster_array[cluster_counter][5] = timestamp_array[counter] - timestamp_array[potential_start]
        cluster_counter += 1 # this is qualified as a cluster, record it
        potential_start = counter # move the potential start forward
    else:
        potential_start = counter # this is not qualified as a cluster, move the potential start forward
    counter += 1

# step 3: export the result
print("there are " + str(cluster_counter) + " clusters in this stream chats. Here are their stats.")
for cluster in cluster_array:
    if (cluster[4] > 0): # some spots in this array are empty, we must exclude them
        print("begin at chat " + str(cluster[0]) + ", " + str(cluster[2]) + " ---- " +
              "end at chat" + str(cluster[1]) + ", " + str(cluster[3]) + " ---- " +
              str(cluster[4]) + " chats within " + str(cluster[5]) + " seconds")