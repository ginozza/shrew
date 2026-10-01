package io.shrew;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Shrew {
    private static final MethodHandle TRAIN_FILE;

    static {
        String[] candidates = {
            "target/release/shrew.dll",
            "../target/release/shrew.dll",
            "../../target/release/shrew.dll",
            "shrew.dll"
        };
        Path dll = null;
        for (String c : candidates) {
            Path p = Paths.get(c).toAbsolutePath().normalize();
            if (Files.exists(p)) {
                dll = p;
                break;
            }
        }
        if (dll != null) {
            System.load(dll.toString());
        } else {
            System.loadLibrary("shrew");
        }

        var lookup = SymbolLookup.loaderLookup();
        TRAIN_FILE = Linker.nativeLinker().downcallHandle(
            lookup.find("shrew_train_file").orElseThrow(),
            FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS
            )
        );
    }

    public record TrainResult(int epochs, double loss) {}

    public static TrainResult train(String swPath) {
        return train(swPath, 1);
    }

    public static TrainResult train(String swPath, int dtype) {
        try (Arena arena = Arena.ofConfined()) {
            Path p = Paths.get(swPath);
            if (!Files.exists(p)) {
                String[] candidates = {
                    swPath,
                    "../" + swPath,
                    "../../" + swPath
                };
                for (String c : candidates) {
                    Path cand = Paths.get(c).toAbsolutePath().normalize();
                    if (Files.exists(cand)) {
                        p = cand;
                        break;
                    }
                }
            }
            MemorySegment pathSeg = arena.allocateFrom(p.toAbsolutePath().toString());
            MemorySegment lossSeg = arena.allocate(ValueLayout.JAVA_DOUBLE);
            int epochs = (int) TRAIN_FILE.invokeExact(pathSeg, dtype, lossSeg);
            if (epochs < 0) {
                throw new RuntimeException("Training failed for " + swPath);
            }
            return new TrainResult(epochs, lossSeg.get(ValueLayout.JAVA_DOUBLE, 0));
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }
}
