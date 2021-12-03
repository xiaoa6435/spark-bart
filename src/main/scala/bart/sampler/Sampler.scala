package bart.sampler

trait Sampler extends Serializable {
  type T
  type A
  val chainId: Int
  var value: T
  def update(data: A): Unit
  def copy(newChainId: Int): Sampler
  def toString: String
}
