import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Solution {

    static int[] solution(int[][] m) {
        Matrix<Integer> stateObservations = Matrix.from(boxIntegerArray(m));

        if (m.length == 0) {
            return new int[]{0, 1};
        }

        if (stateObservations.getNumRows() == 1) {
            return new int[]{1, 1};
        }

        // Remove stable form states (absorbing states)
        List<Integer> stableFormStates = findStableForms(stateObservations);
        stateObservations = stateObservations.removeRows(stableFormStates.stream().mapToInt(i -> i).toArray());

        // Transform our observations into probabilities
        Matrix<Fraction> stateProbabilities = stateObservations.mapRows(row -> {
            long denominator = row.stream().mapToInt(Integer::intValue).sum();
            return row.stream().map(i -> Fraction.create(i, denominator)).collect(Collectors.toList());
        });

        // R matrix
        Matrix<Fraction> stableFormProbabilities = Matrix.empty();

        // Q matrix
        Matrix<Fraction> unstableTransformationProbabilities = Matrix.empty();

        // Build new probability matrices
        for (int i = 0; i < stateProbabilities.getNumColumns(); i++) {
            Matrix<Fraction> sub = stateProbabilities.submatrix(0, i, stateProbabilities.getNumRows(), 1);
            if (stableFormStates.contains(i)) {
                stableFormProbabilities = stableFormProbabilities.concat(sub);
            } else {
                unstableTransformationProbabilities = unstableTransformationProbabilities.concat(sub);
            }
        }

        // fundamental = (I - Q) ^ -1
        Matrix<Fraction> fundamental = invert(Matrix.identity(unstableTransformationProbabilities.getNumRows())
                .mapValues(i -> Fraction.create(i, 1))
                .entrywiseOperation(unstableTransformationProbabilities, Fraction::subtract));

        Matrix<Fraction> FR = fundamental.multiply(stableFormProbabilities, Fraction::multiply, Fraction.toSum());
        List<Fraction> answer = FR.getRow(0);

        long lcd = Fraction.ONE.leastCommonDenominator(answer.toArray(new Fraction[0]));

        // change fractions to use common denominator and concat the lcd for final solution
        return Stream.concat(answer.stream().map(i -> i.getNumerator() * (lcd / i.getDenominator())), Stream.of(lcd))
                .mapToInt(Long::intValue)
                .toArray();
    }

    /**
     * Boxes an array of {@link int} to an array of {@link Integer}.
     *
     * @param array A primitive int array.
     * @return An {@link Integer} array.
     */
    static Integer[][] boxIntegerArray(int[][] array) {
        return Arrays.stream(array).map(inner -> Arrays.stream(inner).boxed().toArray(Integer[]::new)).toArray(Integer[][]::new);
    }

    /**
     * Given a matrix of observed state changes during doomsday fuel production, returns the index of the rows
     * representing stable forms of fuel.
     *
     * @param observations A {@link Matrix} of observed state changes (see readme.txt)
     * @return A {@link List} of row indices.
     */
    static List<Integer> findStableForms(Matrix<Integer> observations) {
        List<Integer> stableFormStates = new ArrayList<>();
        for (int i = 0; i < observations.getNumRows(); i++) {
            if (MathUtil.isZeroVector(observations.getRow(i))) {
                stableFormStates.add(i);
            }
        }
        return stableFormStates;
    }

    /**
     * Calculates the inverse of a {@link Matrix<Fraction>} using Gauss-Jordan elimination.
     * TODO: Write generic inversion function for the Matrix class. This currently works for Fractions.
     *
     * @param matrix The {@link Matrix} to invert.
     * @return The inverse of the specified matrix.
     */
    static Matrix<Fraction> invert(Matrix<Fraction> matrix) {
        if (matrix.getNumColumns() != matrix.getNumRows()) {
            throw new IllegalStateException("Cannot invert a non-square matrix");
        }

        if (matrix.getNumRows() == 1) {
            Fraction[][] arr = new Fraction[][]{{matrix.at(0, 0).inverse()}};
            return Matrix.from(arr);
        }

        if (matrix.getNumRows() == 2) {
            Fraction a = matrix.at(0, 0);
            Fraction b = matrix.at(0, 1);
            Fraction c = matrix.at(1, 0);
            Fraction d = matrix.at(1, 1);

            Fraction determinant = a.multiply(d)
                    .subtract(b.multiply(c))
                    .inverse();

            Matrix<Fraction> mod = matrix.set(0, 0, d)
                    .set(1, 1, a)
                    .set(1, 0, c.negation())
                    .set(0, 1, b.negation());

            return mod.mapValues(i -> i.multiply(determinant));
        }

        // Augmented matrix: specified matrix concatenated with the identity matrix
        Matrix<Fraction> augmented = matrix.concat(
                Matrix.identity(matrix.getNumRows()).mapValues(i -> Fraction.create(i, 1))
        );

        // Find the row with the largest left-most value
        for (int c = 0; c < matrix.getNumColumns(); c++) {
            Fraction maxLeftmost = Fraction.create(Integer.MIN_VALUE, 1);
            int maxLeftmostRow = 0;
            for (int r = 0; r < matrix.getNumRows(); r++) {
                Fraction fraction = augmented.at(r, c);
                if (maxLeftmost.compareTo(fraction) < 0) {
                    maxLeftmost = fraction;
                    maxLeftmostRow = r;
                }
            }

            final Fraction scalar = maxLeftmost.inverse();

            augmented = augmented
                    .swapRows(c, maxLeftmostRow)
                    .mapRow(c, row -> row.stream().map(t -> t.multiply(scalar)).collect(Collectors.toList()));

            for (int r = 0; r < augmented.getNumRows(); r++) {
                Fraction entry = augmented.at(r, c);
                if (!entry.equals(Fraction.ZERO) && r != c) {
                    final List<Fraction> originalRow = augmented.getRow(r);
                    final List<Fraction> mmRow = augmented.getRow(c);
                    final Fraction multiple = entry.negation();

                    List<Fraction> newRow = new ArrayList<>();

                    for (int i = 0; i < originalRow.size(); i++) {
                        newRow.add(originalRow.get(i).add(mmRow.get(i).multiply(multiple)));
                    }

                    augmented = augmented.mapRow(r, row -> newRow);
                }
            }
        }

        return augmented.submatrix(0, matrix.getNumRows(), matrix.getNumRows(), matrix.getNumColumns());
    }

    /**
     * Helper class with some math operations.
     */
    static class MathUtil {

        /**
         * Determines if the specified vector consists solely of zero's.
         *
         * @param vec The {@link List<Integer>} representing the vector.
         * @return Whether or not the specified vector is a zero vector.
         */
        static boolean isZeroVector(List<Integer> vec) {
            for (int i : vec) {
                if (i != 0) {
                    return false;
                }
            }
            return true;
        }

        static Fraction max(Fraction a, Fraction b) {
            return a.compareTo(b) > 0 ? a : b;
        }
    }
}

/**
 * Immutable fraction
 *
 * @author Ugnius Rumsevicius
 */
class Fraction implements Comparable<Fraction> {

    private final long numerator;
    private final long denominator;

    private Fraction(long numerator, long denominator) {
        if (denominator == 0) {
            throw new IllegalArgumentException("Denominator must not be zero");
        }

        long gcd = Math.abs(gcd(numerator, denominator));
        if (denominator < 0) {
            denominator = Math.negateExact(denominator);
            numerator = Math.negateExact(numerator);
        }

        this.numerator = numerator / gcd;
        this.denominator = denominator / gcd;
    }

    public static Fraction create(long numerator, long denominator) {
        if (numerator == 0) {
            return ZERO;
        } else if (numerator == denominator) {
            return ONE;
        } else {
            return new Fraction(numerator, denominator);
        }
    }

    public Fraction add(Fraction fraction) {
        return subtract(fraction.negation());
    }

    public Fraction subtract(Fraction frac) {
        long lcm = lcm(denominator, frac.denominator);
        long a = Math.multiplyExact(numerator, (lcm / denominator));
        long b = Math.multiplyExact(frac.numerator, (lcm / frac.denominator));
        return create(Math.subtractExact(a, b), lcm);
    }

    public Fraction multiply(Fraction scalar) {
        return create(Math.multiplyExact(numerator, scalar.numerator), Math.multiplyExact(denominator, scalar.denominator));
    }

    public Fraction inverse() {
        if (this.equals(ZERO)) {
            return ZERO;
        }
        return create(denominator, numerator);
    }

    public Fraction negation() {
        return create(Math.negateExact(numerator), denominator);
    }

    public long getNumerator() {
        return numerator;
    }

    public long getDenominator() {
        return denominator;
    }

    public long leastCommonDenominator(Fraction fraction) {
        return lcm(this.denominator, fraction.denominator);
    }

    public long leastCommonDenominator(Fraction... fractions) {
        if (fractions.length == 0) {
            throw new IllegalArgumentException("At least one fraction is required to find LCD");
        }
        long denom = fractions[0].getDenominator();
        for (Fraction f : fractions) {
            denom = lcm(denom, f.getDenominator());
        }
        return denom;
    }

    public static Collector<Fraction, FractionSumCollector.FractionAccumulator, Fraction> toSum() {
        return new FractionSumCollector();
    }

    private long gcd(long a, long b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }

    private long lcm(long a, long b) {
        return a * (b / gcd(a, b));
    }

    @Override
    public String toString() {
        return "(" + numerator + "/" + denominator + ")";
    }

    @Override
    public int compareTo(Fraction fraction) {
        long x = Math.subtractExact(Math.multiplyExact(numerator, fraction.denominator), Math.multiplyExact(fraction.numerator, denominator));
        return Long.signum(x);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Fraction fraction = (Fraction) o;
        return numerator == fraction.numerator &&
                denominator == fraction.denominator;
    }

    @Override
    public int hashCode() {
        return Objects.hash(numerator, denominator);
    }

    public static final Fraction ZERO = new Fraction(0, 1);
    public static final Fraction ONE = new Fraction(1, 1);
    public static final Fraction MAX_VALUE = new Fraction(Long.MAX_VALUE, 1);
    public static final Fraction MIN_VALUE = new Fraction(Long.MIN_VALUE, 1);

    private static class FractionSumCollector implements Collector<Fraction, FractionSumCollector.FractionAccumulator, Fraction> {

        private static class FractionAccumulator {
            private Fraction sum;

            FractionAccumulator() {
                sum = Fraction.ZERO;
            }

            FractionAccumulator(Fraction sum) {
                this.sum = sum;
            }

            void update(Fraction fraction) {
                sum = sum.add(fraction);
            }

            Fraction getSum() {
                return sum;
            }
        }

        @Override
        public Supplier<FractionAccumulator> supplier() {
            return FractionAccumulator::new;
        }

        @Override
        public BiConsumer<FractionAccumulator, Fraction> accumulator() {
            return FractionAccumulator::update;
        }

        @Override
        public BinaryOperator<FractionAccumulator> combiner() {
            return (a, b) -> {
                a.update(b.getSum());
                Fraction combined = a.getSum();
                return new FractionAccumulator(combined);
            };
        }

        @Override
        public Function<FractionAccumulator, Fraction> finisher() {
            return FractionAccumulator::getSum;
        }

        @Override
        public Set<Characteristics> characteristics() {
            return new HashSet<>();
        }
    }
}

/**
 * An immutable matrix.
 *
 * @param <T> The type of each entry.
 */
class Matrix<T> {

    private final List<List<T>> matrix;
    private final int numRows;
    private final int numColumns;

    private Matrix(List<List<T>> matrix) {
        if (!verifyRectangular(matrix)) {
            throw new IllegalArgumentException("Matrix must be rectangular");
        }
        this.numRows = matrix.size();
        this.matrix = unmodifiableDeepCopy(matrix);
        if (this.numRows == 0) {
            this.numColumns = 0;
            return;
        }

        this.numColumns = matrix.get(0).size();
    }

    /**
     * Generates a {@link Matrix} from a specified two-dimensional array.
     *
     * @param matrix The two dimensional array.
     * @param <T>    Type of each entry.
     * @return A {@link Matrix} built from the specified array.
     */
    public static <T> Matrix<T> from(T[][] matrix) {
        return new Matrix<>(toList(matrix));
    }

    /**
     * Generates an identity matrix of the specified size.
     *
     * @param size The size of the identity matrix.
     * @return An identity matrix.
     */
    public static Matrix<Integer> identity(int size) {
        List<List<Integer>> identity = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            List<Integer> row = new ArrayList<>();
            for (int j = 0; j < size; j++) {
                row.add(0);
            }
            row.set(i, 1);
            identity.add(row);
        }
        return new Matrix<>(identity);
    }

    /**
     * Retrieves the element in the matrix at the specified row and column.
     *
     * @param r The row of the entry.
     * @param c The column of the entry.
     * @return The entry at the specified location.
     */
    public T at(int r, int c) {
        validate(r, c);
        return matrix.get(r).get(c);
    }

    /**
     * Sets the specified entry to a new value and returns the resulting matrix.
     *
     * @param r     The row index of the entry.
     * @param c     The column index of the entry.
     * @param value The new value of the entry.
     * @return A new {@link Matrix} with the modified entry.
     */
    public Matrix<T> set(int r, int c, T value) {
        List<List<T>> copy = deepCopy(matrix);
        copy.get(r).set(c, value);
        return new Matrix<>(copy);
    }

    /**
     * Retrieves the row of the matrix specified by its index.
     *
     * @param r The index of the row.
     * @return A {@link List} representing the row.
     */
    public List<T> getRow(int r) {
        return new ArrayList<>(matrix.get(r));
    }

    /**
     * Retrieves the column of the matrix specified by its index.
     *
     * @param c The index of the column.
     * @return A {@link List} representing the column.
     */
    public List<T> getColumn(int c) {
        List<T> col = new ArrayList<>();
        for (List<T> row : this.matrix) {
            col.add(row.get(c));
        }
        return col;
    }

    /**
     * Performs a specified entrywise operation onto this matrix using a given matrix.
     *
     * @param matrix The {@link Matrix} to operate with.
     * @param op     The operation to perform between the two matrices the results in the new value.
     * @param <U>    The type of the entries in the specified matrix.
     * @param <R>    The type of the entries in the resulting matrix.
     * @return A new {@link Matrix} as a result of the operations.
     */
    public <U, R> Matrix<R> entrywiseOperation(Matrix<U> matrix, BiFunction<T, U, R> op) {
        if (getNumRows() != matrix.getNumRows() || getNumColumns() != matrix.getNumColumns()) {
            throw new IllegalArgumentException("Matrices dimensions must match for entrywise operations");
        }

        List<List<R>> copy = new ArrayList<>();

        for (int r = 0; r < this.matrix.size(); r++) {
            List<R> row = new ArrayList<>();
            for (int c = 0; c < this.matrix.get(r).size(); c++) {
                row.add(op.apply(this.matrix.get(r).get(c), matrix.matrix.get(r).get(c)));
            }
            copy.add(row);
        }

        return new Matrix<>(copy);
    }

    /**
     * Performs matrix multiplication.
     *
     * @param matrix       The matrix to multiply by.
     * @param multiplyOp   Specifies how to perform the multiplication operation.
     * @param sumCollector A {@link Collector} to use to achieve a sum of the entries.
     * @param <U>          The type of entries of the specified matrix.
     * @param <A>          The type of the accumulator used to sum the entries.
     * @param <R>          The type of the entries of the resulting matrix.
     * @return The resulting product {@link Matrix}.
     */
    public <U, A, R> Matrix<R> multiply(Matrix<U> matrix, BiFunction<T, U, R> multiplyOp, Collector<R, A, R> sumCollector) {
        if (getNumColumns() != matrix.getNumRows()) {
            throw new IllegalArgumentException("Incompatible matrix for multiplication");
        }

        List<List<R>> result = new ArrayList<>();

        for (int r = 0; r < getNumRows(); r++) {
            List<R> resultRow = new ArrayList<>();
            for (int c = 0; c < matrix.getNumColumns(); c++) {
                List<T> row = getRow(r);
                List<U> col = matrix.getColumn(c);
                A accumulator = sumCollector.supplier().get();
                for (int i = 0; i < row.size(); i++) {
                    sumCollector.accumulator().accept(accumulator, multiplyOp.apply(row.get(i), col.get(i)));
                }
                resultRow.add(sumCollector.finisher().apply(accumulator));
            }
            result.add(resultRow);
        }
        return new Matrix<>(result);
    }

    /**
     * Calculates a submatrix specified by the top left corner and size.
     *
     * @param startR Top left corner row.
     * @param startC Top left corner column.
     * @param r      Number of rows in the submatrix.
     * @param c      Number of columns in the submatrix.
     * @return A {@link Matrix} which is a subset of this matrix.
     */
    public Matrix<T> submatrix(int startR, int startC, int r, int c) {
        validate(startR + r - 1, startC + c - 1);
        List<List<T>> sub = new ArrayList<>();
        for (int i = startR; i < startR + r; i++) {
            List<T> subRow = new ArrayList<>();
            for (int j = startC; j < startC + c; j++) {
                subRow.add(at(i, j));
            }
            sub.add(subRow);
        }
        return new Matrix<>(sub);
    }

    /**
     * Swaps two rows and returns the resulting matrix.
     *
     * @param r1 The first row.
     * @param r2 The second row.
     * @return A new {@link Matrix} with the specified rows swapped.
     */
    public Matrix<T> swapRows(int r1, int r2) {
        List<List<T>> copy = deepCopy(this.matrix);
        List<T> temp = copy.get(r1);
        copy.set(r1, copy.get(r2));
        copy.set(r2, temp);
        return new Matrix<>(copy);
    }

    /**
     * Removes all specified rows and returns the resulting matrix.
     *
     * @param rows Indices of the rows to remove.
     * @return A new {@link Matrix} without the specified rows.
     */
    public Matrix<T> removeRows(int... rows) {
        List<List<T>> matrixCopy = new ArrayList<>();

        for (int i = 0; i < matrix.size(); i++) {
            if (Arrays.stream(rows).boxed().collect(Collectors.toList()).contains(i)) {
                continue;
            }

            matrixCopy.add(matrix.get(i));
        }

        return new Matrix<>(matrixCopy);
    }

    /**
     * Concatenates two matrices and returns the resulting matrix.
     * For this to be possible, both matrices must have the same number of rows.
     *
     * @param matrix The {@link Matrix} to concatentate on this matrix.
     * @return A new {@link Matrix} formed by concatenation.
     */
    public Matrix<T> concat(Matrix<T> matrix) {
        if (this.matrix.size() == 0) {
            return new Matrix<>(matrix.matrix);
        }
        if (numRows != matrix.getNumRows()) {
            throw new IllegalArgumentException("Incompatable column sizes for concatenation");
        }

        List<List<T>> copy = deepCopy(this.matrix);
        for (int i = 0; i < matrix.getNumRows(); i++) {
            copy.get(i).addAll(matrix.getRow(i));
        }

        return new Matrix<>(copy);
    }

    /**
     * Changes each entry in the matrix as specified and returns the resulting matrix.
     *
     * @param mapper The transformation function.
     * @param <R>    The type of entries of the resulting matrix.
     * @return A new transformed {@link Matrix}.
     */
    public <R> Matrix<R> mapValues(Function<T, R> mapper) {
        List<List<R>> matrix_ = new ArrayList<>();
        for (List<T> row : this.matrix) {
            List<R> row_ = new ArrayList<>();
            for (T t : row) {
                row_.add(mapper.apply(t));
            }
            matrix_.add(row_);
        }
        return new Matrix<>(matrix_);
    }

    /**
     * Changes each row in the matrix as specified and returns the resulting matrix.
     *
     * @param mapper The transformation function.
     * @param <R>    The type of the entries of the resulting matrix.
     * @return A new transformed {@link Matrix}.
     */
    public <R> Matrix<R> mapRows(Function<List<T>, List<R>> mapper) {
        List<List<R>> matrix_ = new ArrayList<>();
        for (List<T> row : this.matrix) {
            matrix_.add(mapper.apply(row));
        }
        return new Matrix<>(matrix_);
    }

    /**
     * Changes the specified row as specified and returns the resulting matrix.
     *
     * @param r      The index of the row to transform.
     * @param mapper The transformation function.
     * @return A new transformed {@link Matrix}.
     */
    public Matrix<T> mapRow(int r, Function<List<T>, List<T>> mapper) {
        List<List<T>> matrix_ = new ArrayList<>();
        for (int i = 0; i < getNumRows(); i++) {
            if (i == r) {
                List<T> row = mapper.apply(new ArrayList<>(this.matrix.get(i)));
                matrix_.add(row);
                continue;
            }
            matrix_.add(this.matrix.get(i));

        }
        return new Matrix<>(matrix_);
    }

    /**
     * Returns an empty matrix.
     *
     * @param <T> The type of the entries in the matrix.
     * @return An empty {@link Matrix}.
     */
    @SuppressWarnings("unchecked")
    public static <T> Matrix<T> empty() {
        return EMPTY;
    }

    /**
     * Checks where the row and column are within the bounds of the matrix.
     * If not, throws an {@link IndexOutOfBoundsException}.
     *
     * @param r The row index.
     * @param c The column index.
     */
    private void validate(int r, int c) {
        if (r < 0 || c < 0 || r >= matrix.size() || c >= matrix.get(r).size()) {
            throw new IndexOutOfBoundsException();
        }
    }

    /**
     * Checks if the matrix is a rectangular shape (non-jagged).
     *
     * @param matrix The matrix represented by a {@link List<List<T>>}.
     * @return Whether the matrix is rectangular.
     */
    private boolean verifyRectangular(List<List<T>> matrix) {
        int i = -1;
        for (List<T> t : matrix) {
            if (i == -1) {
                i = t.size();
                continue;
            }

            if (i != t.size()) {
                return false;
            }

            i = t.size();
        }
        return true;
    }

    /**
     * Returns a two dimensional {@link List} constructed from the specified two-dimensional array.
     *
     * @param arr The array to build from.
     * @param <T> The type of the entries in the array.
     * @return A two-dimensional {@link List} consisting of entries from the specified array.
     */
    private static <T> List<List<T>> toList(T[][] arr) {
        List<List<T>> m = new ArrayList<>();
        for (T[] t : arr) {
            m.add(Arrays.stream(t).collect(Collectors.toList()));
        }
        return m;
    }

    /**
     * Creates a deep copy of the specified two-dimensional {@link List}.
     *
     * @param matrix The {@link List} to copy.
     * @param <T>    The type of the entries in the list.
     * @return A copy of the specified list.
     */
    private static <T> List<List<T>> deepCopy(List<List<T>> matrix) {
        List<List<T>> copy = new ArrayList<>();
        for (List<T> row : matrix) {
            List<T> rowCopy = new ArrayList<>(row);
            copy.add(rowCopy);
        }
        return copy;
    }

    /**
     * Creates an immutable deep copy of the specified two-dimensional {@link List}.
     *
     * @param matrix The {@link List} to copy.
     * @param <T>    The type of the entries in the list.
     * @return An immutable copy of the specified list.
     */
    private static <T> List<List<T>> unmodifiableDeepCopy(List<List<T>> matrix) {
        List<List<T>> copy = new ArrayList<>();
        for (List<T> row : matrix) {
            List<T> rowCopy = new ArrayList<>(row);
            copy.add(Collections.unmodifiableList(rowCopy));
        }
        return Collections.unmodifiableList(copy);
    }

    public int getNumRows() {
        return numRows;
    }

    public int getNumColumns() {
        return numColumns;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[\n");
        for (List<T> arr : matrix) {
            builder.append("  ");
            builder.append(arr.toString());
            builder.append("\n");
        }
        builder.append("]");
        return builder.toString();
    }

    @SuppressWarnings("rawtypes")
    private static final Matrix EMPTY = new Matrix<>(new ArrayList<>());
}
