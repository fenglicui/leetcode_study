package com;

import javafx.util.Pair;

import java.util.HashMap;
import java.util.Map;

class UndergroundSystem {

    // 记录乘客进站信息
    Map<Integer, Pair<String, Integer>> userMap;
    // 记录startStation-endStation的乘车记录，value为< 总乘车时间, 乘车次数>
    Map<String, Pair<Integer, Integer>> timeMap;

    public UndergroundSystem() {
        userMap = new HashMap<>();
        timeMap = new HashMap<>();
    }

    public void checkIn(int id, String stationName, int t) {
        userMap.put(id, new Pair<>(stationName, t));
    }

    public void checkOut(int id, String stationName, int t) {
        // 出站会产生乘车记录，不再记录出站信息，直接记录乘车记录
        String startStation = userMap.get(id).getKey();
        int startTime = userMap.get(id).getValue();
        String log = startStation + stationName;
        Pair<Integer, Integer> pair = timeMap.getOrDefault(log, new Pair<>(0, 0));
        // 更新总乘车时间和乘车次数
        timeMap.put(log, new Pair<>(pair.getKey() + (t - startTime), pair.getValue() + 1));
    }

    public double getAverageTime(String startStation, String endStation) {
        double time = timeMap.get(startStation + endStation).getKey();
        int count = timeMap.get(startStation + endStation).getValue();
        return time / count;
    }
}