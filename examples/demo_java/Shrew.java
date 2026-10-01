import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Paths;

public class Shrew {
    private static final MethodHandle TRAIN;

    static {
        System.load(Paths.get("target/release/shrew.dll").toAbsolutePath().toString());
        var lookup = SymbolLookup.loaderLookup();
        TRAIN = Linker.nativeLinker().downcallHandle(
            lookup.find("shrew_train_file").orElseThrow(),
            FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS
            )
        );
    }

    public record Result(int epochs, double loss) {}

    public static Result train(String swPath) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment loss = arena.allocate(ValueLayout.JAVA_DOUBLE);
            int epochs = (int) TRAIN.invokeExact(arena.allocateFrom(swPath), 1, loss);
            return new Result(epochs, loss.get(ValueLayout.JAVA_DOUBLE, 0));
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }
}
