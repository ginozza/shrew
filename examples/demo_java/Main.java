import io.shrew.Shrew;

public class Main {
    public static void main(String[] args) {
        var res = Shrew.train("examples/model.sw");
        System.out.printf("Epochs: %d, Loss: %.6f%n", res.epochs(), res.loss());
    }
}
