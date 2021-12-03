package bart

import scala.collection.parallel.mutable.ParArray
import scala.reflect.ClassTag

trait ParArrOrSeqArr[T] extends Serializable{
  def map[B](f: T => B)(implicit ct: ClassTag[B]): ParArrOrSeqArr[B]
  def filter(p: T => Boolean): ParArrOrSeqArr[T]
  def foreach[U](f: T => U): Unit
  def reduce[U >: T](op: (U, U) ⇒ U): U
  def partition(pred: T ⇒ Boolean): (ParArrOrSeqArr[T], ParArrOrSeqArr[T])
  def length: Int
  def aggregate[S](z: ⇒ S)(seqop: (S, T) ⇒ S, combop: (S, S) ⇒ S): S
  def head: T
}

class SeqArr[T](val df: Array[T]) extends ParArrOrSeqArr[T] {
  def map[B](f: T => B)(implicit ct: ClassTag[B]) = new SeqArr(df.map(f))
  def filter(p: T => Boolean) = new SeqArr(df.filter(p))
  def foreach[U](f: T => U): Unit = df.foreach(f)
  def reduce[U >: T](op: (U, U) ⇒ U): U = df.reduce(op)
  def partition(pred: T ⇒ Boolean): (SeqArr[T], SeqArr[T]) = {
    val (df1, df2) = df.partition(pred)
    (new SeqArr(df1), new SeqArr(df2))
  }
  def length: Int = df.length

  def aggregate[S](z: ⇒ S)(seqop: (S, T) ⇒ S, combop: (S, S) ⇒ S): S = df.aggregate(z)(seqop, combop)
  def head: T = df.head
}

class ParArr[T](val df: ParArray[T]) extends ParArrOrSeqArr[T] {
  def map[B](f: T => B)(implicit ct: ClassTag[B]) = new ParArr(df.map(f))
  def filter(p: T => Boolean) = new ParArr(df.filter(p))
  def foreach[U](f: T => U): Unit = df.foreach(f)
  def reduce[U >: T](op: (U, U) ⇒ U): U = df.reduce(op)
  def partition(pred: T ⇒ Boolean): (ParArr[T], ParArr[T]) = {
    val (df1, df2) = df.partition(pred)
    (new ParArr(df1), new ParArr(df2))
  }
  def length: Int = df.length
  def aggregate[S](z: ⇒ S)(seqop: (S, T) ⇒ S, combop: (S, S) ⇒ S): S = df.aggregate(z)(seqop, combop)
  def head: T = df.head
}
