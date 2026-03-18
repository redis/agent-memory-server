package com.redis.agentmemory.models.longtermemory;

/**
 * Result from a memory search operation.
 */
public class MemoryRecordResult extends MemoryRecord {

    private double dist;
    private Double score;
    private String scoreType;

    public MemoryRecordResult() {
        super();
    }

    public double getDist() {
        return dist;
    }

    public void setDist(double dist) {
        this.dist = dist;
    }

    public Double getScore() {
        return score;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public String getScoreType() {
        return scoreType;
    }

    @com.fasterxml.jackson.annotation.JsonProperty("score_type")
    public void setScoreType(String scoreType) {
        this.scoreType = scoreType;
    }

    @Override
    public String toString() {
        return "MemoryRecordResult{" +
                "dist=" + dist +
                ", score=" + score +
                ", scoreType='" + scoreType + '\'' +
                ", " + super.toString() +
                '}';
    }
}
