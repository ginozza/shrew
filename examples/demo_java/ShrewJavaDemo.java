import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ShrewJavaDemo {
    public static void main(String[] args) throws Throwable {
        System.out.println("=================================================");
        System.out.println("  Shrew Java 25 (Project Panama / FFM) Demo");
        System.out.println("=================================================");

        // Load shrew.dll
        Path libPath = Paths.get("target/release/shrew.dll").toAbsolutePath();
        System.out.println("Loading Shrew native library: " + libPath);
        System.load(libPath.toString());

        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.loaderLookup();

        // 1. shrew_version
        MethodHandle versionFn = linker.downcallHandle(
            lookup.find("shrew_version").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.ADDRESS)
        );
        MemorySegment versionSeg = (MemorySegment) versionFn.invokeExact();
        String version = versionSeg.reinterpret(64).getString(0);
        System.out.println("Shrew Engine Version: " + version);

        // 2. Hardware detection
        MethodHandle cudaFn = linker.downcallHandle(
            lookup.find("shrew_cuda_is_available").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_INT)
        );
        int cudaAvailable = (int) cudaFn.invokeExact();
        System.out.println("CUDA Available in C API: " + (cudaAvailable == 1));

        // 3. Methods for Tensor operations
        MethodHandle fromDataFn = linker.downcallHandle(
            lookup.find("shrew_tensor_from_data").orElseThrow(),
            FunctionDescriptor.of(
                ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG,
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT
            )
        );

        MethodHandle matmulFn = linker.downcallHandle(
            lookup.find("shrew_tensor_matmul").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
        );

        MethodHandle toDataFn = linker.downcallHandle(
            lookup.find("shrew_tensor_to_data").orElseThrow(),
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
        );

        MethodHandle printFn = linker.downcallHandle(
            lookup.find("shrew_tensor_print").orElseThrow(),
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
        );

        MethodHandle freeFn = linker.downcallHandle(
            lookup.find("shrew_tensor_free").orElseThrow(),
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
        );

        try (Arena arena = Arena.ofConfined()) {
            // Matrix A [2, 2] = [[1.0, 2.0], [3.0, 4.0]]
            MemorySegment dataA = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, 1.0, 2.0, 3.0, 4.0);
            MemorySegment shapeA = arena.allocateFrom(ValueLayout.JAVA_LONG, 2L, 2L);
            MemorySegment tensorA = (MemorySegment) fromDataFn.invokeExact(dataA, shapeA, 2L, 1, 0); // F64, CPU

            // Matrix B [2, 2] = [[5.0, 6.0], [7.0, 8.0]]
            MemorySegment dataB = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, 5.0, 6.0, 7.0, 8.0);
            MemorySegment shapeB = arena.allocateFrom(ValueLayout.JAVA_LONG, 2L, 2L);
            MemorySegment tensorB = (MemorySegment) fromDataFn.invokeExact(dataB, shapeB, 2L, 1, 0);

            // Matrix multiplication C = A @ B
            // Expected:
            // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
            MemorySegment tensorC = (MemorySegment) matmulFn.invokeExact(tensorA, tensorB);

            System.out.println("\nResult of Matrix Multiplication C = A @ B computed by Shrew:");
            printFn.invokeExact(tensorC);

            MemorySegment outData = arena.allocate(ValueLayout.JAVA_DOUBLE, 4L);
            int count = (int) toDataFn.invokeExact(tensorC, outData, 4L);
            System.out.print("Extracted " + count + " elements into Java memory: [ ");
            for (int i = 0; i < 4; i++) {
                System.out.print(outData.getAtIndex(ValueLayout.JAVA_DOUBLE, i) + " ");
            }
            System.out.println("]");

            double c0 = outData.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
            double c1 = outData.getAtIndex(ValueLayout.JAVA_DOUBLE, 1);
            double c2 = outData.getAtIndex(ValueLayout.JAVA_DOUBLE, 2);
            double c3 = outData.getAtIndex(ValueLayout.JAVA_DOUBLE, 3);
            if (c0 == 19.0 && c1 == 22.0 && c2 == 43.0 && c3 == 50.0) {
                System.out.println("\n>>> SUCCESS: Shrew successfully executed directly from Java! <<<\n");
            }

            // Free tensors
            freeFn.invokeExact(tensorA);
            freeFn.invokeExact(tensorB);
            freeFn.invokeExact(tensorC);
        }
    }
}
