package de.cogmod.utilities;

/**
 * @author Sebastian Otte
 */
public interface LearningListener {
    public void afterEpoch(final int epoch, final double... performanceValues);
}