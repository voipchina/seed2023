#/bin/bash
TAG=registry.cn-hangzhou.aliyuncs.com/mario_odyssey/seed2023
NOW=`date +%m%d%H%M`
# NOW=11060022
docker build --squash -t $TAG:$NOW .

