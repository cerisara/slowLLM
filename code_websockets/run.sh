trap 'kill $BGPID; exit' INT
ssh -N olkihost -R 9487:152.81.7.89:5000 &
BGPID=$!
sleep 2
source /home/xtof/envs/transformers/bin/activate
python main.py &
sleep 2
# for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
#     python client.py $i &
#     sleep 1
# done
python client.py 0

kill $BGPID
exit

