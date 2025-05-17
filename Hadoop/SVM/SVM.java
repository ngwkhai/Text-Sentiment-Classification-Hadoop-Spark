import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.Counters;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.regex.Pattern;

public class SVM {
    public static enum Global_Counters {
        TRUE_POSITIVE,
        FALSE_POSITIVE,
        TRUE_NEGATIVE,
        FALSE_NEGATIVE,
        SAMPLES_PROCESSED
    }

    // Hyperparameters
    private static final int EPOCHS = 5;
    private static final double INITIAL_LR = 0.1;
    private static final double REGULARIZATION = 0.01;

    // Shared text cleaning regex
    private static final Pattern URL_REGEX = Pattern.compile("(?i)(https?:\\/\\/\\S+)");
    private static final Pattern NON_ALPHA = Pattern.compile("[^a-zA-Z ]");

    public static class Map_Training extends Mapper<Object, Text, Text, DoubleWritable> {
        private Map<String, Double> weights = new HashMap<>();
        private double eta;

        @Override
        protected void setup(Context context) {
            // load current weights if available
            eta = INITIAL_LR / (1 + context.getConfiguration().getInt("epoch", 1) * REGULARIZATION);
            Path modelPath = new Path("training/model_epoch_" + context.getConfiguration().getInt("epoch", 1));
            try {
                FileSystem fs = FileSystem.get(context.getConfiguration());
                if (fs.exists(modelPath)) {
                    for (FileStatus f : fs.listStatus(modelPath)) {
                        if (!f.isFile()) continue;
                        FSDataInputStream in = fs.open(f.getPath());
                        BufferedReader br = new BufferedReader(new InputStreamReader(in));
                        String line;
                        while ((line = br.readLine()) != null) {
                            String[] parts = line.split("\t");
                            weights.put(parts[0], Double.parseDouble(parts[1]));
                        }
                        br.close();
                    }
                }
            } catch (IOException e) {
                // ignore, start from zeros
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] cols = value.toString().split(",");
            if (cols.length < 4) return;
            int label = cols[1].equals("1") ? 1 : -1;
            String text = cols[3].toLowerCase();
            text = URL_REGEX.matcher(text).replaceAll(" ");
            text = NON_ALPHA.matcher(text).replaceAll(" ");
            text = text.replaceAll("\\s+", " ").trim();
            StringTokenizer itr = new StringTokenizer(text);
            double dot = 0.0;
            while (itr.hasMoreTokens()) {
                String w = itr.nextToken();
                dot += weights.getOrDefault(w, 0.0);
            }
            if (label * dot < 1) {
                for (Map.Entry<String, Double> e : weights.entrySet()) {
                    context.write(new Text(e.getKey()), new DoubleWritable(-eta * REGULARIZATION * e.getValue()));
                }
                itr = new StringTokenizer(text);
                while (itr.hasMoreTokens()) {
                    String w = itr.nextToken();
                    context.write(new Text(w), new DoubleWritable(eta * label));
                }
            }
        }
    }

    public static class Reduce_Training extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            double sum = 0.0;
            for (DoubleWritable v : values) sum += v.get();
            context.write(key, new DoubleWritable(sum));
        }
    }

    public static class Map_Testing extends Mapper<Object, Text, Text, Text> {
        private Map<String, Double> weights = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException {
            FileSystem fs = FileSystem.get(context.getConfiguration());
            // load final epoch model
            Path modelPath = new Path("training/model_epoch_" + EPOCHS);
            for (FileStatus f : fs.listStatus(modelPath)) {
                if (!f.isFile()) continue;
                FSDataInputStream in = fs.open(f.getPath());
                BufferedReader br = new BufferedReader(new InputStreamReader(in));
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split("\t");
                    weights.put(parts[0], Double.parseDouble(parts[1]));
                }
                br.close();
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] cols = value.toString().split(",");
            if (cols.length < 4) return;
            String id = cols[0];
            int trueLabel = cols[1].equals("1") ? 1 : -1;
            String text = cols[3].toLowerCase();
            text = URL_REGEX.matcher(text).replaceAll(" ");
            text = NON_ALPHA.matcher(text).replaceAll(" ");
            text = text.replaceAll("\\s+", " ").trim();
            double score = 0.0;
            StringTokenizer itr = new StringTokenizer(text);
            while (itr.hasMoreTokens()) {
                score += weights.getOrDefault(itr.nextToken(), 0.0);
            }
            int pred = score >= 0 ? 1 : -1;
            if (pred == 1 && trueLabel == 1) context.getCounter(Global_Counters.TRUE_POSITIVE).increment(1);
            if (pred == 1 && trueLabel == -1) context.getCounter(Global_Counters.FALSE_POSITIVE).increment(1);
            if (pred == -1 && trueLabel == -1) context.getCounter(Global_Counters.TRUE_NEGATIVE).increment(1);
            if (pred == -1 && trueLabel == 1) context.getCounter(Global_Counters.FALSE_NEGATIVE).increment(1);
            context.getCounter(Global_Counters.SAMPLES_PROCESSED).increment(1);
            context.write(new Text(id + "@" + text), new Text(pred == 1 ? "POSITIVE" : "NEGATIVE"));
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: SVM <train_dirs> <test_dirs> <train_split> <test_split>");
            System.exit(1);
        }
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path outBase = new Path("training");
        if (fs.exists(outBase)) fs.delete(outBase, true);

        // iterative epochs
        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            conf.setInt("epoch", epoch);
            Path inModel = (epoch == 1) ? null : new Path("training/model_epoch_" + (epoch - 1));
            Path outModel = new Path("training/model_epoch_" + epoch);
            if (fs.exists(outModel)) fs.delete(outModel, true);
            Job trainJob = Job.getInstance(conf, "SVM-Training-Epoch-" + epoch);
            trainJob.setJarByClass(SVM.class);
            trainJob.setMapperClass(Map_Training.class);
            trainJob.setReducerClass(Reduce_Training.class);
            trainJob.setMapOutputKeyClass(Text.class);
            trainJob.setMapOutputValueClass(DoubleWritable.class);
            trainJob.setOutputKeyClass(Text.class);
            trainJob.setOutputValueClass(DoubleWritable.class);
            for (String dir : args[0].split(",")) {
                TextInputFormat.addInputPath(trainJob, new Path(dir));
            }
            long trainSplit = Long.parseLong(args[2]);
            TextInputFormat.setMaxInputSplitSize(trainJob, trainSplit);
            TextOutputFormat.setOutputPath(trainJob, outModel);
            trainJob.waitForCompletion(true);
        }

        // Testing
        Path outputDir = new Path("output");
        if (fs.exists(outputDir)) fs.delete(outputDir, true);
        Job testJob = Job.getInstance(conf, "SVM-Testing");
        testJob.setJarByClass(SVM.class);
        testJob.setMapperClass(Map_Testing.class);
        testJob.setNumReduceTasks(0);
        testJob.setMapOutputKeyClass(Text.class);
        testJob.setMapOutputValueClass(Text.class);
        testJob.setOutputKeyClass(Text.class);
        testJob.setOutputValueClass(Text.class);
        for (String dir : args[1].split(",")) {
            TextInputFormat.addInputPath(testJob, new Path(dir));
        }
        long testSplit = Long.parseLong(args[3]);
        TextInputFormat.setMaxInputSplitSize(testJob, testSplit);
        TextOutputFormat.setOutputPath(testJob, outputDir);
        testJob.waitForCompletion(true);

        // Metrics
        Counters ctr = testJob.getCounters();
        long TP = ctr.findCounter(Global_Counters.TRUE_POSITIVE).getValue();
        long FP = ctr.findCounter(Global_Counters.FALSE_POSITIVE).getValue();
        long TN = ctr.findCounter(Global_Counters.TRUE_NEGATIVE).getValue();
        long FN = ctr.findCounter(Global_Counters.FALSE_NEGATIVE).getValue();

        long total = TP + TN + FP + FN;
        double accuracy = ((double)(TP + TN)) / total;
        double precision = TP / (double)(TP + FP);
        double recall = TP / (double)(TP + FN);
        double f1_score = 2 * precision * recall / (precision + recall);

        System.out.println("Confusion Matrix:");
        System.out.println("TP: " + TP + "\tFP: " + FP);
        System.out.println("FN: " + FN + "\tTN: " + TN);
        System.out.printf("Accuracy: %.4f\n", accuracy);
        System.out.printf("F1-Score: %.4f\n", f1_score);
        System.out.printf("Precision: %.4f\n", precision);
        System.out.printf("Recall: %.4f\n", recall);
    }
}
