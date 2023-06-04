plugins {
    id("java")
    id("org.openjfx.javafxplugin").version("0.0.13")
}

group = "hb"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

javafx {
    version = "17"
    modules("javafx.controls", "javafx.fxml")
}

dependencies {
    implementation("me.tongfei:progressbar:0.9.5")
    implementation("org.controlsfx:controlsfx:11.1.2")
}

task("runTraining", JavaExec::class) {
    mainClass.set("hb.app.Training")
    classpath = sourceSets["main"].runtimeClasspath
}

task("runTesting", JavaExec::class) {
    mainClass.set("hb.app.Testing")
    classpath = sourceSets["main"].runtimeClasspath
}

task("runDrawUI", JavaExec::class) {
    mainClass.set("hb.app.DrawUI")
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs("--module-path", classpath.asPath, "--add-modules", "javafx.controls,javafx.fxml")
}
