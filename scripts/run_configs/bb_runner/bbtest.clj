(require '[clojure.java.shell :refer [sh]]
         '[clojure.core.async :as async :refer [<! >! go chan close! alts! alts!!]]
         '[babashka.fs :as fs]
         '[taoensso.timbre :as log :refer [info error debug]]
         '[babashka.cli :as cli]
         '[lambdaisland.cli :as cli]
         '[clojure.pprint :refer [pprint]]
         '[malli.core :as m]
         '[malli.instrument :as mi]
         '[malli.dev.pretty :as pretty]
         '[clojure.string :as str])

(comment
    (require '[malli.dev :as dev])

    ;; It's main entry points is dev/start!, taking same options as
    ;; mi/instrument!. It runs mi/instrument! and mi/collect! (for all loaded
    ;; namespaces) once and starts watching the function registry for changes.
    ;; Any change that matches the filters will cause automatic
    ;; re-instrumentation for the functions. dev/stop! removes all
    ;; instrumentation and stops watching the registry.
    (dev/start!)
    (dev/stop!)
)

(def small-int [:int {:max 6}])
(m/=> square [:=> [:cat small-int] :int])
(defn square [x] (* x x))

(m/=> test-multi-arity 
    ; https://www.wedesoft.de/software/2023/12/25/clojure-function-schemas-with-malli/
    [:function [:=> [:cat :double] :string]
               [:=> [:cat :double :double] :string]])
(defn test-multi-arity
    ([x] (str "one arg: " x))
    ([x y] (str "two args: " x ", " y)))

(defn examples-stuff 
    "collection of examples to do with babashka and core.async"
    {:flags ["-v, --verbose" "Increases verbosity"
             "--input FILE" {:doc "Specify the input file"
                             :default "default.txt"}
             "--env=<dev|prod|staging>" {:doc "Select an environment"
                                         :default "dev"
                                         :parse keyword}]}
    [opts]

    (println (str "Current file path is: " *file*))

    (println "Passed options:")
    (pprint opts)

    (-> (sh "ls" "-la") 
        (println))


    (info "Hello, this is bbtest.clj running!")

    ;; just using (go) is good. Uses a thread pool. use (thread) for dedicated thread
    (let [c1 (chan)
        c2 (chan)]
    (go (while true
            (let [[v ch] (alts! [c1 c2])]
            (println "Read" v "from" ch))))
    (go (>! c1 "hi"))
    (go (>! c2 "there")))

    (let [c (chan)]
        (go (>! c (sh "amlt" "--version")))

        (info "is go blocking? this should print first")

        (let [[data] (alts!! [c])]
            (println "AMLT TL Command Output:")
            (println (:out data)))
    )

    (async/<!! 
        (go
        (let [c (chan)]
            (async/thread
            (let [result (sh "amlt" "tl")]
                (>! c result)))
            
            (info "Running AMLT TL command asynchronously...")

            (let [result (<! c)]
            (println "AMLT TL Command Output:")
            (println (:out result)))))
    )
)




(mi/collect!)
(mi/instrument! {:report (pretty/thrower)})


(square 6)
(test-multi-arity 0.0 0.0 0.0) 


(cli/dispatch #'examples-stuff)
