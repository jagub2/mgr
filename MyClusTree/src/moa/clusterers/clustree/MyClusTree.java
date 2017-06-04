/*
 *    ClusTree.java
 *    Copyright (C) 2010 RWTH Aachen University, Germany
 *    @author Sanchez Villaamil (moa@cs.rwth-aachen.de)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 *
 */

package moa.clusterers.clustree;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.clustree.util.Budget;
import moa.clusterers.clustree.util.SimpleBudget;
import moa.core.Measurement;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

/**
 * Citation: ClusTree: Philipp Kranen, Ira Assent, Corinna Baldauf, Thomas Seidl:
 * The ClusTree: indexing micro-clusters for anytime stream mining.
 * Knowl. Inf. Syst. 29(2): 249-272 (2011)
 */
public class MyClusTree extends AbstractClusterer {
    private static final long serialVersionUID = 1L;

    public IntOption horizonOption = new IntOption("horizon",
            'h', "Range of the window.", 1000);

    public IntOption maxHeightOption = new IntOption(
            "maxHeight", 'H',
            "The maximal height of the tree", getDefaultHeight());

    public FlagOption breadthFirstStrategyOption = new FlagOption(
            "breadthFirstStrategy", 'B',
            "Use breadth first strategy");

    /* START: Hierarchical concept drift detector */
    public IntOption windowSizeOption = new IntOption("windowSize",
            'w', "Window size of concept drift detector.", 200);

    public FloatOption detectionThresholdOption = new FloatOption("detectionThreshold",
            'd', "Threshold when sample is detected as novelty. Values 0.0-1.0", 0.3);

    public FloatOption conceptDriftThresholdOption = new FloatOption("conceptDriftThreshold", 'c',
            "Threshold of how many % of conceptDriftWindow is treated as concept drift", 0.3);

    public FlagOption hierarchicalConceptDriftDetectionOption = new FlagOption(
            "hierarchicalConceptDriftDetection", 'D',
            "Use hierarchical concept drift detector");

    protected int windowSize;
    protected ArrayList<Double> conceptDriftWindow;
    protected ArrayList<Instance> conceptDriftWindowSamples;
    protected int conceptDriftWindowPointer = 0;
    protected double detectionThreshold;
    protected double conceptDriftThreshold;
    protected long startOfWindowTimestamp = 0;
    protected boolean hierarchicalConceptDriftDetection;

    private boolean isLearningAfterDrift = false;
    /* END: Hierarchical concept drift detector */

    /* START: Hierarchical adaptation algorithm */
    public FloatOption usageThresholdOption = new FloatOption("usageThreshold", 'u',
            "Threshold of how many % of branches is used", 0.3);

    public IntOption insertionsBetweenAdaptationOption = new IntOption("insertionsBetweenAdaptation", 'i',
            "How many insertions before pruning unused branches", 2000);

    public FlagOption hierarchicalAdaptationOption = new FlagOption(
            "hierarchicalAdaptation", 'A',
            "Use hierarchical concept drift detector");

    protected double usageThreshold;
    protected long insertionsBetweenAdaptation;
    protected boolean hierarchicalAdaptation;

    private int numberInsertionsAdapt; // duh

    protected ArrayList<Double[]> adaptationWindow;
    /* END: Hierarchical adaptation algorithm */

    protected int getDefaultHeight() {
        return 8;
    }

    private static int INSERTIONS_BETWEEN_CLEANUPS = 10000;
    /**
     * The root node of the tree.
     */
    protected MyNode root;
    // Information about the data represented in this tree.
    /**
     * Dimensionality of the data points managed by this tree.
     */
    private int numberDimensions;
    /**
     * Parameter for the weighting function use to weight the entries.
     */
    protected double negLambda;
    /**
     * The current height of the tree. Should always be smaller than maxHeight.
     */
    private int height;
    /**
     * The maximal height of the tree.
     */
    protected int maxHeight;
    /**
     * This variable is used to keep the inverse height that is stored in every
     * node correct.
     */
    private int numRootSplits;
    /**
     * The threshold for the weighting of an Entry. An Entry is irrelevant, if
     * it is in a leaf and the weightedN of the data Cluster is smaller than
     * this threshold.
     *
     * @see Entry#data
     */
    private double weightThreshold = 0.05;
    /**
     * Number of points inserted into the tree.
     */
    private int numberInsertions;
    private long timestamp;

    /**
     * Parameter to determine wich strategy to use
     */
    protected boolean breadthFirstStrat = false;

    //TODO: cleanup
    private MyEntry alsoUpdate;

    public MyClusTree() {
        super();
        conceptDriftWindow = new ArrayList<>();
        conceptDriftWindowSamples = new ArrayList<>();
        //hierarchicalConceptDriftDetectionOption.set();
        adaptationWindow = new ArrayList<>();
    }

    @Override
    public void resetLearningImpl() {
        breadthFirstStrat = breadthFirstStrategyOption.isSet();
        negLambda = (1.0 / (double) horizonOption.getValue())
                * (Math.log(weightThreshold) / Math.log(2));
        maxHeight = maxHeightOption.getValue();
        numberDimensions = -1;
        root = null;
        timestamp = 0;
        height = 0;
        numRootSplits = 0;
        numberInsertions = 0;

        /* START: Hierarchical concept drift detector */
        windowSize = windowSizeOption.getValue();
        detectionThreshold = detectionThresholdOption.getValue();
        conceptDriftThreshold = conceptDriftThresholdOption.getValue();
        conceptDriftWindowPointer = 0;
        conceptDriftWindow.clear();
        conceptDriftWindowSamples.clear();
        startOfWindowTimestamp = 0;
        hierarchicalConceptDriftDetection = hierarchicalConceptDriftDetectionOption.isSet();
        /* END: Hierarchical concept drift detector */

        /* START: Hierarchical adaptation algorithm */
        usageThreshold = usageThresholdOption.getValue();
        insertionsBetweenAdaptation = insertionsBetweenAdaptationOption.getValue();
        hierarchicalAdaptation = hierarchicalAdaptationOption.isSet();
        numberInsertionsAdapt = 0;
        adaptationWindow.clear();
        /* END: Hierarchical adaptation algorithm */
    }


    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public double[] getVotesForInstance(Instance inst) {
        return null;
    }

    @Override
    public boolean implementsMicroClusterer() {
        return true;
    }


    @Override
    public void trainOnInstanceImpl(Instance instance) {
        if(conceptDriftWindowSamples.size() == 0 && isLearningAfterDrift) {
            isLearningAfterDrift = false;
        }
        timestamp++;

        //TODO check if instance contains label
        if (root == null) {
            numberDimensions = instance.numAttributes();
            root = new MyNode(numberDimensions, 0);
        } else {
            if (numberDimensions != instance.numAttributes())
                System.out.println("Wrong dimensionality, expected:" + numberDimensions + "found:" + instance.numAttributes());
        }

        if(hierarchicalConceptDriftDetection) {
            conceptDriftWindow.add(new Double(0));
            conceptDriftWindowSamples.add(instance.copy());
        }

        ClusKernel newPointAsKernel = new ClusKernel(instance.toDoubleArray(), numberDimensions);
        insert(newPointAsKernel, new SimpleBudget(1000), timestamp);
        if(hierarchicalConceptDriftDetection && !isLearningAfterDrift) {
            conceptDriftWindow.set(conceptDriftWindowPointer, conceptDriftWindow.get(conceptDriftWindowPointer) / (this.height + 1.0));
            conceptDriftWindowPointer++;

            if (conceptDriftWindow.size() >= windowSize) {
                if (isConceptDriftPresentInWindow()) {
                    maxHeight = maxHeightOption.getValue();
                    numberDimensions = -1;
                    root = null;
                    height = 0;
                    numRootSplits = 0;
                    numberInsertions = 0;
                    numberInsertionsAdapt = 0;

                    isLearningAfterDrift = true;
                    for(int i = conceptDriftWindowSamples.size() - 1; i >= 0; i--) {
                        trainOnInstanceImpl(conceptDriftWindowSamples.get(i));
                        conceptDriftWindowSamples.remove(i);
                    }
                }
                conceptDriftWindowPointer = 0;
                conceptDriftWindow.clear();
                conceptDriftWindowSamples.clear();
                startOfWindowTimestamp = timestamp;
            }
        }
        if(hierarchicalAdaptation && !isLearningAfterDrift) {
            adaptationWindow.add(Arrays.stream(instance.toDoubleArray()).boxed().toArray(Double[]::new));
        }
    }


    /**
     * Insert a new point in the <code>Tree</code>. The point should be
     * represented as a cluster with a single data point(i.e. N = 1). A
     * <code>Budget</code> class is also given, which is informed of the number
     * of operation the tree does, and informs the tree when it does not have
     * time left and should stop the insertion.
     *
     * @param newPoint  The point to be inserted.
     * @param budget    The budget and statistics recollector for the insertion.
     * @param timestamp The moment at which this point is inserted.
     * @see Kernel
     * @see Budget
     */
    public void insert(ClusKernel newPoint, Budget budget, long timestamp) {
        if (breadthFirstStrat) {
            insertBreadthFirst(newPoint, budget, timestamp);
        } else {
            MyEntry rootEntry = new MyEntry(this.numberDimensions,
                    root, timestamp, null, null);
            ClusKernel carriedBuffer = new ClusKernel(this.numberDimensions);
            MyEntry toInsertHere = insert(newPoint, carriedBuffer, root, rootEntry,
                    budget, timestamp);

            if (toInsertHere != null) {
                //toInsertHere.incrementUsage();
                this.numRootSplits++;
                this.height += this.height < this.maxHeight ? 1 : 0;

                MyNode newRoot = new MyNode(this.numberDimensions,
                        toInsertHere.getChild().getRawLevel() + 1);
                newRoot.addEntry(rootEntry, timestamp);
                newRoot.addEntry(toInsertHere, timestamp);
                rootEntry.setNode(newRoot);
                toInsertHere.setNode(newRoot);
                MyEntry parentEntry = toInsertHere.getParentEntry();
                /*while(parentEntry != null) {
                    parentEntry.incrementUsage();
                    parentEntry = parentEntry.getParentEntry();
                }*/
                this.root = newRoot;
            }
        }

        numberInsertions++;
        numberInsertionsAdapt++;
        if(numberInsertions % INSERTIONS_BETWEEN_CLEANUPS == 0) {
            cleanUp(root, 0);
            numberInsertions = 0;
        }
        if(hierarchicalAdaptation && numberInsertionsAdapt % insertionsBetweenAdaptation == 0) {
            numberInsertionsAdapt = 0;
            ArrayList<MyEntry> entriesToUpdate = getNodesToUpdateUsage();
            updateUsages(entriesToUpdate, false);
            removeUnusedEntries(root);
            updateUsages(entriesToUpdate, true);
            adaptationWindow.clear();
        }
    }

    private void updateUsages(ArrayList<MyEntry> entries, boolean reset) {
        for(MyEntry entry: entries) {
            if(entry == null) {
                continue;
            }
            if(reset) {
                entry.resetUsage();
            } else {
                entry.updateUsage(1);
            }
            MyEntry parent = entry.getParentEntry();
            while(parent != null) {
                if(reset) {
                    parent.resetUsage();
                } else {
                    parent.updateUsage(1);
                }
                parent = parent.getParentEntry();
            }
        }
    }

    private ArrayList<MyEntry> getNodesToUpdateUsage() {
        int level = this.height;
        ArrayList<MyEntry> entriesToUpdate = new ArrayList<>();

        LinkedList<MyNode> nodeQueue = new LinkedList<>();

        nodeQueue.add(root);

        while (!nodeQueue.isEmpty()) {
            MyNode current = nodeQueue.remove();
            int currentLevel = current.getLevel(this);
            boolean isLeaf = (current.isLeaf() && currentLevel <= maxHeight)
                    || currentLevel == maxHeight;

            if (currentLevel == level) {
                assert (currentLevel <= maxHeight);

                MyEntry[] entries = current.getEntries();
                for (int i = 0; i < entries.length; i++) {
                    MyEntry entry = entries[i];
                    if (entry.isEmpty()) {
                        continue;
                    }
                    entriesToUpdate.add(entry);
                }
            } else if (!current.isLeaf()) {
                MyEntry[] entries = current.getEntries();
                for (int i = 0; i < entries.length; i++) {
                    MyEntry entry = entries[i];

                    if (entry.isEmpty()) {
                        continue;
                    }

                    nodeQueue.add(entry.getChild());
                }
            }
        }
        return entriesToUpdate;
    }

    private ArrayList<Integer> getMinMaxUsagesOnLevel(int level) {
        ArrayList<Integer> usages = new ArrayList<>(); // poor man's tuple
        usages.add(new Integer(Integer.MAX_VALUE));
        usages.add(new Integer(Integer.MIN_VALUE));

        LinkedList<MyNode> nodeQueue = new LinkedList<>();

        nodeQueue.add(root);

        while (!nodeQueue.isEmpty()) {
            MyNode current = nodeQueue.remove();
            if(current == null) {
                continue;
            }
            int currentLevel = current.getLevel(this);
            boolean isLeaf = (current.isLeaf() && currentLevel <= maxHeight)
                    || currentLevel == maxHeight;

            if (currentLevel == level) {
                assert (currentLevel <= maxHeight);

                MyEntry[] entries = current.getEntries();
                for (int i = 0; i < entries.length; i++) {
                    MyEntry entry = entries[i];
                    if (entry.isEmpty()) {
                        continue;
                    }
                    if(entry.getUsage() > usages.get(1)) {
                        usages.set(1, entry.getUsage());
                    }
                    if(entry.getUsage() < usages.get(0)) {
                        usages.set(0, entry.getUsage());
                    }
                }
            } else if (!current.isLeaf()) {
                MyEntry[] entries = current.getEntries();
                for (int i = 0; i < entries.length; i++) {
                    MyEntry entry = entries[i];

                    if (entry.isEmpty()) {
                        continue;
                    }

                    nodeQueue.add(entry.getChild());
                }
            }
        }
        return usages;
    }

    private void removeUnusedEntries(MyNode node) {
        if(node != null || !node.isLeaf()) {
            MyEntry[] entries = node.getEntries();
            for(int i = 0; i < entries.length; i++) {
                if(entries[i].getChild() != null) {
                    removeUnusedEntries(entries[i].getChild());
                }
                ArrayList<Integer> minMax = getMinMaxUsagesOnLevel(node.getLevel(this));
                double coefficient = (minMax.get(0) + minMax.get(1) + adaptationWindow.size()) / 3.0;
                if(((double)entries[i].getUsage()/coefficient) < usageThreshold && !entries[i].isEmpty()) {
                    entries[i].getData().clear();
                    entries[i].getData().setWeight(0);
                }
            }
            node.setEntries(entries);
        }
    }

    /**
     * insert newPoint into the tree using the BreadthFirst strategy, i.e.: insert into
     * the closest entry in a leaf node.
     *
     * @param newPoint
     * @param budget
     * @param timestamp
     * @return
     */
    private MyEntry insertBreadthFirst(ClusKernel newPoint, Budget budget, long timestamp) {
        //check all leaf nodes and get the one with the closest entry to newPoint
        MyNode bestFit = findBestLeafNode(newPoint);
        bestFit.makeOlder(timestamp, negLambda);
        MyEntry parent = bestFit.getEntries()[0].getParentEntry();
        // Search for an Entry with a weight under the threshold.
        MyEntry irrelevantEntry = bestFit.getIrrelevantEntry(this.weightThreshold);
        int numFreeEntries = bestFit.numFreeEntries();
        MyEntry newEntry = new MyEntry(newPoint.getCenter().length,
                newPoint, timestamp, parent, bestFit);
        //if there is space, add it to the node ( doesn't ever occur, since nodes are created with 3 entries)
        if (numFreeEntries > 0) {
            bestFit.addEntry(newEntry, timestamp);
        }
        //if outdated cluster in this best fitting node, replace it
        else if (irrelevantEntry != null) {
            irrelevantEntry.overwriteOldEntry(newEntry);
        }
        //if there is space/outdated cluster on path to top, split. Else merge without split
        else {
            if (existsOutdatedEntryOnPath(bestFit) || !this.hasMaximalSize()) {
                // We have to split.
                insertHereWithSplit(newEntry, bestFit, timestamp);
            } else {
                mergeEntryWithoutSplit(bestFit, newEntry,
                        timestamp);
            }
        }
        //update all nodes on path to top.
        if (bestFit.getEntries()[0].getParentEntry() != null)
            updateToTop(bestFit.getEntries()[0].getParentEntry().getNode());
        return null;
    }

    /**
     * This method checks if there is an outdated (or empty) entry on the path from node to root.
     * It updates the weights of nodes on path and then checks if it is outdated.
     *
     * @param node
     * @return true if an outdated/empty entry exists on the path
     */
    private boolean existsOutdatedEntryOnPath(MyNode node) {
        if (node == root) {
            node.makeOlder(timestamp, negLambda);
            return node.getIrrelevantEntry(this.weightThreshold) != null;
        }
        do {
            node = node.getEntries()[0].getParentEntry().getNode();
            node.makeOlder(timestamp, negLambda);
            for (MyEntry e : node.getEntries()) {
                e.recalculateData();
            }
            if (node.numFreeEntries() > 0)
                return true;
            if (node.getIrrelevantEntry(this.weightThreshold) != null)
                return true;
        } while (node.getEntries()[0].getParentEntry() != null);
        return false;
    }

    /**
     * recalculates data for all entries, that lie on the path from the root to the
     * Entry toUpdate.
     */
    private void updateToTop(MyNode toUpdate) {
        while (toUpdate != null) {
            for (MyEntry e : toUpdate.getEntries())
                e.recalculateData();
            if (toUpdate.getEntries()[0].getParentEntry() == null)
                break;
            toUpdate = toUpdate.getEntries()[0].getParentEntry().getNode();
        }
    }

    /**
     * Method called by insertBreadthFirst.
     *
     * @param toInsert
     * @param insertNode
     * @param timestamp
     * @return
     */
    private Entry insertHereWithSplit(MyEntry toInsert, MyNode insertNode,
                                      long timestamp) {
        //Handle root split
        if (insertNode.getEntries()[0].getParentEntry() == null) {
            root.makeOlder(timestamp, negLambda);
            MyEntry irrelevantEntry = insertNode.getIrrelevantEntry(this.weightThreshold);
            int numFreeEntries = insertNode.numFreeEntries();
            if (irrelevantEntry != null) {
                irrelevantEntry.overwriteOldEntry(toInsert);
            } else if (numFreeEntries > 0) {
                insertNode.addEntry(toInsert, timestamp);
            } else {
                this.numRootSplits++;
                this.height += this.height < this.maxHeight ? 1 : 0;
                MyEntry oldRootEntry = new MyEntry(this.numberDimensions,
                        root, timestamp, null, null);
                MyNode newRoot = new MyNode(this.numberDimensions,
                        this.height);
                MyEntry newRootEntry = split(toInsert, root, oldRootEntry, timestamp);
                newRoot.addEntry(oldRootEntry, timestamp);
                newRoot.addEntry(newRootEntry, timestamp);
                this.root = newRoot;
                for (MyEntry c : oldRootEntry.getChild().getEntries())
                    c.setParentEntry(root.getEntries()[0]);
                for (MyEntry c : newRootEntry.getChild().getEntries())
                    c.setParentEntry(root.getEntries()[1]);
            }
            return null;
        }
        insertNode.makeOlder(timestamp, negLambda);
        MyEntry irrelevantEntry = insertNode.getIrrelevantEntry(this.weightThreshold);
        int numFreeEntries = insertNode.numFreeEntries();
        if (irrelevantEntry != null) {
            irrelevantEntry.overwriteOldEntry(toInsert);
        } else if (numFreeEntries > 0) {
            insertNode.addEntry(toInsert, timestamp);
        } else {
            // We have to split.
            MyEntry parentEntry = insertNode.getEntries()[0].getParentEntry();
            MyEntry residualEntry = split(toInsert, insertNode, parentEntry, timestamp);
            if (alsoUpdate != null) {
                alsoUpdate = residualEntry;
            }
            MyNode nodeForResidualEntry = insertNode.getEntries()[0].getParentEntry().getNode();
            //recursive call
            return insertHereWithSplit(residualEntry, nodeForResidualEntry, timestamp);
        }

        //no Split
        return null;
    }


    // XXX: Document the insertion when the final implementation is done.
    private MyEntry insertHere(MyEntry newEntry, MyNode currentNode,
                               MyEntry parentEntry, ClusKernel carriedBuffer, Budget budget,
                               long timestamp) {

        int numFreeEntries = currentNode.numFreeEntries();

        //printTree(this.root, 0);
        //System.out.println("=====");

        // Insert the buffer that we carry.
        if (!carriedBuffer.isEmpty()) {
            MyEntry bufferEntry = new MyEntry(this.numberDimensions,
                    carriedBuffer, timestamp, parentEntry, currentNode);

            if (numFreeEntries <= 1) {
                // Distance from buffer to entries.
                MyEntry nearestEntryToCarriedBuffer =
                        currentNode.nearestEntry(newEntry);
                double distanceNearestEntryToBuffer =
                        nearestEntryToCarriedBuffer.calcDistance(newEntry);

                // Distance between buffer and point to insert.
                double distanceBufferNewEntry =
                        newEntry.calcDistance(carriedBuffer);

                // Best distance between Entries in the Node.
                BestMergeInNode bestMergeInNode =
                        calculateBestMergeInNode(currentNode);

                // See what the minimal distance is and do the correspoding
                // action.
                if (distanceNearestEntryToBuffer <= distanceBufferNewEntry
                        && distanceNearestEntryToBuffer <= bestMergeInNode.distance) {
                    // Aggregate buffer entry to nearest entry in node.
                    nearestEntryToCarriedBuffer.aggregateEntry(bufferEntry,
                            timestamp, this.negLambda);
                } else if (distanceBufferNewEntry <= distanceNearestEntryToBuffer
                        && distanceBufferNewEntry <= bestMergeInNode.distance) {
                    newEntry.mergeWith(bufferEntry);
                } else {
                    currentNode.mergeEntries(bestMergeInNode.entryPos1,
                            bestMergeInNode.entryPos2);
                    currentNode.addEntry(bufferEntry, timestamp);
                }

            } else {
                assert (currentNode.isLeaf());
                currentNode.addEntry(bufferEntry, timestamp);
            }
        }

        // Normally the insertion of the carries buffer does not change the
        // number of free entries, but in case of future changes we calculate
        // the number again.
        numFreeEntries = currentNode.numFreeEntries();

        // Search for an Entry with a weight under the threshold.
        MyEntry irrelevantEntry = currentNode.getIrrelevantEntry(this.weightThreshold);
        if (currentNode.isLeaf() && irrelevantEntry != null) {
            irrelevantEntry.overwriteOldEntry(newEntry);
        } else if (numFreeEntries >= 1) {
            currentNode.addEntry(newEntry, timestamp);
        } else {
            if (currentNode.isLeaf() && (this.hasMaximalSize()
                    || !budget.hasMoreTime())) {
                mergeEntryWithoutSplit(currentNode, newEntry,
                        timestamp);
            } else {
                // We have to split.
                return split(newEntry, currentNode, parentEntry, timestamp);
            }
        }

        return null;
    }

    /**
     * This method calculates the distances between the new point and each Entry in a leaf node.
     * It returns the node that contains the entry with the smallest distance
     * to the new point.
     *
     * @param newPoint
     * @return best fitting node
     */
    private MyNode findBestLeafNode(ClusKernel newPoint) {
        double minDist = Double.MAX_VALUE;
        MyNode bestFit = null;
        for (MyNode e : collectLeafNodes(root)) {
            if (newPoint.calcDistance(e.nearestEntry(newPoint).getData()) < minDist) {
                bestFit = e;
                minDist = newPoint.calcDistance(e.nearestEntry(newPoint).getData());
            }
        }
        if (bestFit != null)
            return bestFit;
        else
            return root;
    }

    private ArrayList<MyNode> collectLeafNodes(MyNode curr) {
        ArrayList<MyNode> toReturn = new ArrayList<>();
        if (curr == null)
            return toReturn;
        if (curr.isLeaf()) {
            toReturn.add(curr);
            return toReturn;
        } else {
            for (MyEntry e : curr.getEntries())
                toReturn.addAll(collectLeafNodes(e.getChild()));
            return toReturn;
        }
    }

    private double getConceptDriftInWindow() {
        double count = 0;
        for(int i = 0; i < conceptDriftWindow.size(); i++) {
            //if(conceptDriftWindow.get(i) > detectionThreshold) {
                //count += 1.0;
            //}
            count += conceptDriftWindow.get(i);
        }
        count /= conceptDriftWindow.size();
        //return count / (double) conceptDriftWindow.size();
        return count;
    }

    private boolean isConceptDriftPresentInWindow() {
        /*for(int i = 0; i < conceptDriftWindow.size(); i++) {
            System.out.print(conceptDriftWindow.get(i) + " ");
        }
        System.out.println();*/
        return getConceptDriftInWindow() > conceptDriftThreshold;
    }

    private boolean isSampleFitting(ClusKernel existingKernel, ClusKernel sample) {
        double value = 0;
        for(int i = 0; i < existingKernel.getCenter().length; i++) {
            value += Math.pow(existingKernel.getCenter()[i] - sample.getCenter()[i], 2);
        }
        return value < Math.pow(existingKernel.getRadius(), 2);
    }

    // TODO: Expand all function that work on entries to work with the Budget.
    private MyEntry insert(ClusKernel pointToInsert, ClusKernel carriedBuffer,
                           MyNode currentNode, MyEntry parentEntry, Budget budget, long timestamp) {
        assert (currentNode != null);
        assert (currentNode.isLeaf()
                || currentNode.getEntries()[0].getChild() != null);

        currentNode.makeOlder(timestamp, this.negLambda);

        // This variable will be changed from to null to an actual reference
        // in the following if-else block if we have to insert something here,
        // either because this is a leaf, or because of split propagation.
        MyEntry toInsertHere = null;

        MyEntry bestEntry = null;

        if (currentNode.isLeaf()) {
            // At the end of the function the entry will be inserted.
            toInsertHere = new MyEntry(this.numberDimensions,
                    pointToInsert, timestamp, parentEntry, currentNode);
        } else {
            bestEntry = currentNode.nearestEntry(pointToInsert);
            bestEntry.aggregateCluster(pointToInsert, timestamp,
                    this.negLambda);

            boolean isCarriedBufferEmpty = carriedBuffer.isEmpty();

            MyEntry bestBufferEntry = null;
            if (!isCarriedBufferEmpty) {
                bestBufferEntry = currentNode.nearestEntry(carriedBuffer);
                bestBufferEntry.aggregateCluster(carriedBuffer, timestamp,
                        this.negLambda);
            }

            if (!budget.hasMoreTime()) {
                bestEntry.aggregateToBuffer(pointToInsert, timestamp,
                        this.negLambda);
                if (!isCarriedBufferEmpty) {
                    bestBufferEntry.aggregateToBuffer(carriedBuffer,
                            timestamp, this.negLambda);
                }
                return null;
            }

            // If the way of the buffer differs from the way of the point to
            // be inserted, leave the buffer here.
            if (!isCarriedBufferEmpty && (bestEntry != bestBufferEntry)) {
                bestBufferEntry.aggregateToBuffer(carriedBuffer, timestamp,
                        this.negLambda);
                carriedBuffer.clear();
            }
            // Take the buffer of the best entry for the point to be inserted
            // along.
            ClusKernel takeAlongBuffer = bestEntry.emptyBuffer(timestamp,
                    this.negLambda);
            carriedBuffer.add(takeAlongBuffer);

            // Recursive call.
            toInsertHere = insert(pointToInsert, carriedBuffer,
                    bestEntry.getChild(), bestEntry, budget, timestamp);
        }

        // If the above block has a new Entry for this place insert it.
        if (toInsertHere != null) {
            if(bestEntry != null && hierarchicalConceptDriftDetection) {
                if (!isSampleFitting(bestEntry.getData(), pointToInsert)
                        || (!bestEntry.getNode().isLeaf() && (bestEntry.getTimestamp() > startOfWindowTimestamp))) {
                    conceptDriftWindow.set(conceptDriftWindowPointer, conceptDriftWindow.get(conceptDriftWindowPointer) + 1);
                }
            }
            /*if(hierarchicalAdaptation) {
                toInsertHere.incrementUsage();
            }*/

            return this.insertHere(toInsertHere, currentNode, parentEntry,
                    carriedBuffer, budget, timestamp);
        }

        // If nothing else needs to be done in all the above levels
        // return null to signalize it.
        return null;
    }

    /**
     * Inserts an <code>Entry</code> into a <code>Node</code> without inducing
     * a split.
     *
     * @param node      The node at which the entry is to be inserted.
     * @param newEntry  The entry to be inserted.
     * @param timestamp The moment at which this occurs.
     */
    private void mergeEntryWithoutSplit(MyNode node, MyEntry newEntry, long timestamp) {

        MyEntry nearestEntryToCarriedBuffer =
                node.nearestEntry(newEntry);
        double distanceNearestEntryToBuffer =
                nearestEntryToCarriedBuffer.calcDistance(newEntry);

        BestMergeInNode bestMergeInNode =
                calculateBestMergeInNode(node);

        if (distanceNearestEntryToBuffer < bestMergeInNode.distance) {
            nearestEntryToCarriedBuffer.aggregateEntry(newEntry, timestamp,
                    this.negLambda);
        } else {
            node.mergeEntries(bestMergeInNode.entryPos1,
                    bestMergeInNode.entryPos2);
            node.addEntry(newEntry, timestamp);
        }
    }

    /**
     * Calculates the best merge possible between two nodes in a node. This
     * means that the pair with the smallest distance is found.
     *
     * @param node The node in which these two entries have to be found.
     * @return An object which encodes the two position of the entries with the
     * smallest distance in the node and the distance between them.
     * @see BestMergeInNode
     * @see Entry#calcDistance(tree.Entry)
     */
    private BestMergeInNode calculateBestMergeInNode(MyNode node) {
        assert (node.numFreeEntries() == 0);

        MyEntry[] entries = node.getEntries();

        int toMerge1 = -1;
        int toMerge2 = -1;
        double distanceBetweenMergeEntries = Double.NaN;

        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < entries.length; i++) {
            MyEntry e1 = entries[i];
            for (int j = i + 1; j < entries.length; j++) {
                MyEntry e2 = entries[j];
                double distance = e1.calcDistance(e2);
                if (distance < minDistance) {
                    toMerge1 = i;
                    toMerge2 = j;
                    distanceBetweenMergeEntries = distance;
                }
            }
        }

        assert (toMerge1 != -1 && toMerge2 != -1);
        if (Double.isNaN(distanceBetweenMergeEntries)) {
            throw new RuntimeException("The minimal distance between two "
                    + "Entrys in a Node was Double.MAX_VAUE. That can hardly "
                    + "be right.");
        }

        return new BestMergeInNode(toMerge1, toMerge2,
                distanceBetweenMergeEntries);
    }

    private boolean hasMaximalSize() {
        // TODO: Improve hasMaximalSize(). For now it just works somehow for testing.
        return this.height == this.maxHeight;
    }

    /**
     * Performs a (2,2) split on the given node with the given entry. This
     * implementation only works if the nodes have three entries each. The split
     * will generate two new nodes. One of them will be put where the old node
     * was, and for the other a new <code>Entry</code> will be generated and
     * returned.
     *
     * @param newEntry    The entry to be added to the node.
     * @param node        The node that is going to be splitted.
     * @param parentEntry The entry in the tree that points at the node that
     *                    is going to be splitted.
     * @param timestamp   The moment at which this split occurs.
     * @return An entry which points at the second node created in the split.
     * This entry has to be introduced later in the tree.
     */
    private MyEntry split(MyEntry newEntry, MyNode node, MyEntry parentEntry,
                          long timestamp) {
        // The implemented split function only works in trees where node
        // have three entries.
        // Splitting only makes sense on full nodes.
        assert (node.numFreeEntries() == 0);
        assert (parentEntry.getChild() == node);

        // All the entries we have to separate in two nodes.
        MyEntry[] allEntries = new MyEntry[4];
        MyEntry[] nodeEntries = node.getEntries();
        for (int i = 0; i < nodeEntries.length; i++) {
            allEntries[i] = new MyEntry(nodeEntries[i]);
        }
        allEntries[3] = newEntry;

        // Clear the given node, since we are going to refill it later.
        node = new MyNode(this.numberDimensions, node.getRawLevel());

        // Calculate the distance of all the possible pairings, since we want
        // to do a (2,2) split.
        double select01 = allEntries[0].calcDistance(allEntries[1])
                + allEntries[2].calcDistance(allEntries[3]);

        double select02 = allEntries[0].calcDistance(allEntries[2])
                + allEntries[1].calcDistance(allEntries[3]);

        double select03 = allEntries[0].calcDistance(allEntries[3])
                + allEntries[1].calcDistance(allEntries[2]);

        // See which of the pairings is minimal and distribute the entries
        // accordingly.
        MyNode residualNode = new MyNode(this.numberDimensions,
                node.getRawLevel());
        if (select01 < select02) {
            if (select01 < select03) {//select01 smallest
                node.addEntry(allEntries[0], timestamp);
                node.addEntry(allEntries[1], timestamp);
                residualNode.addEntry(allEntries[2], timestamp);
                residualNode.addEntry(allEntries[3], timestamp);
            } else {//select03 smallest
                node.addEntry(allEntries[0], timestamp);
                node.addEntry(allEntries[3], timestamp);
                residualNode.addEntry(allEntries[1], timestamp);
                residualNode.addEntry(allEntries[2], timestamp);
            }
        } else {
            if (select02 < select03) {//select02 smallest
                node.addEntry(allEntries[0], timestamp);
                node.addEntry(allEntries[2], timestamp);
                residualNode.addEntry(allEntries[1], timestamp);
                residualNode.addEntry(allEntries[3], timestamp);
            } else {//select03 smallest
                node.addEntry(allEntries[0], timestamp);
                node.addEntry(allEntries[3], timestamp);
                residualNode.addEntry(allEntries[1], timestamp);
                residualNode.addEntry(allEntries[2], timestamp);
            }
        }

        // Set the other node into the tree.
        parentEntry.setChild(node);
        parentEntry.recalculateData();
        int count = 0;
        for (MyEntry e : node.getEntries()) {
            e.setParentEntry(parentEntry);
            if (e.getData().getN() != 0)
                count++;
        }
        //System.out.println(count);
        // Generate a new entry for the residual node.
        MyEntry residualEntry = new MyEntry(this.numberDimensions,
                residualNode, timestamp, parentEntry, node);
        count = 0;
        for (MyEntry e : residualNode.getEntries()) {
            e.setParentEntry(residualEntry);
            if (e.getData().getN() != 0)
                count++;
        }
        //System.out.println(count);
        return residualEntry;
    }

    /**
     * Return the number of time the tree has grown in size. If the tree grows
     * and is then cutted from a certain depth, it also counts.
     *
     * @return The number of times the root node was splitted.
     */
    public int getNumRootSplits() {
        return numRootSplits;
    }

    /**
     * Return the current height of the tree. This should never be greater than
     * <code>maxHeight</code>.
     *
     * @return The height of the tree.
     * @see #maxHeight
     */
    public int getHeight() {
        assert (height <= maxHeight);
        return height;
    }

    private void cleanUp(MyNode currentNode, int level) {
        if (currentNode == null) {
            return;
        }

        MyEntry[] entries = currentNode.getEntries();
        if (level == this.maxHeight) {
            for (int i = 0; i < entries.length; i++) {
                MyEntry e = entries[i];
                e.setChild(null);
            }
        } else {
            for (int i = 0; i < entries.length; i++) {
                MyEntry e = entries[i];
                cleanUp(e.getChild(), level + 1);
            }
        }
    }

    /**
     * @param currentTime The current time
     * @return The kernels at the leaf level as a clustering
     */
    //TODO: Microcluster unter dem Threshhold nich zur�ckgeben (WIe bei outdated entries)
    @Override
    public Clustering getMicroClusteringResult() {
        return getClustering(timestamp, -1);
    }

    @Override
    public Clustering getClusteringResult() {
        return null;
    }


    /**
     * @param currentTime The current time
     * @return The kernels at the given level as a clustering.
     */
    public Clustering getClustering(long currentTime, int targetLevel) {
        if (root == null) {
            return null;
        }

        Clustering clusters = new Clustering();
        LinkedList<MyNode> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            MyNode current = queue.remove();
             if (current == null)
             	continue;
            int currentLevel = current.getLevel(this);
            boolean isLeaf = (current.isLeaf() && currentLevel <= maxHeight)
                    || currentLevel == maxHeight;

            if (currentLevel == targetLevel
                    || (targetLevel == -1 && isLeaf)) {
                assert (currentLevel <= maxHeight);

                MyEntry[] entries = current.getEntries();
                for (int i = 0; i < entries.length; i++) {
                    MyEntry entry = entries[i];
                    if (entry == null || entry.isEmpty()) {
                        continue;
                    }
                    // XXX
                    entry.makeOlder(currentTime, this.negLambda);
                    if (entry.isIrrelevant(this.weightThreshold))
                        continue;

                    ClusKernel gaussKernel = new ClusKernel(entry.getData());

//                  long diff = currentTime - entry.getTimestamp();
//                    if (diff > 0) {
//                        gaussKernel.makeOlder(diff, negLambda);
//                    }

                    clusters.add(gaussKernel);
                }
            } else if (!current.isLeaf()) {
                MyEntry[] entries = current.getEntries();
                for (int i = 0; i < entries.length; i++) {
                    MyEntry entry = entries[i];

                    if (entry.isEmpty()) {
                        continue;
                    }

                    if (entry.isIrrelevant(weightThreshold)) {
                        continue;
                    }

                    queue.add(entry.getChild());
                }
            }
        }

        return clusters;
    }


    /**************************************************************************
     * LOCAL CLASSES
     **************************************************************************/
    /**
     * A class to code the return value of searching the smallest merge in a
     * node.
     */
    class BestMergeInNode {

        /**
         * The position of the first entry in the array of the node.
         */
        public int entryPos1;
        /**
         * The position of the second entry in the array of the node.
         */
        public int entryPos2;
        /**
         * The distance between the two entries.
         */
        public double distance;

        /**
         * The constructor of this return value. It will automatically make
         * sure that the first position is the smaller one of the two.
         *
         * @param pos1     One of the position.
         * @param pos2     One of the position.
         * @param distance The distance between the entries at these positions.
         */
        public BestMergeInNode(int pos1, int pos2,
                               double distance) {
            assert (pos1 != pos2);

            this.distance = distance;

            if (pos1 < pos2) {
                this.entryPos1 = pos1;
                this.entryPos2 = pos2;
            } else {
                this.entryPos1 = pos2;
                this.entryPos2 = pos1;
            }
        }
    }

}
